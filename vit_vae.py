import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Genotype_Induced_Drug_Design.PVAE.utils import celoss


class ViTVAE(nn.Module):
    """
    1D ViT-based VAE that patchifies DNA and Gene expression vectors (2 channels),
    runs attention, and reconstructs both. Compatible with CNNVAE API.
    """

    def __init__(
        self,
        input_dim: int = 15703,
        z_dim: int = 64,
        num_classes: int = 2,
        patch_size: int = 128,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Compute number of patches and padding length
        self.n_patches = (input_dim + patch_size - 1) // patch_size
        self.pad_len = self.n_patches * patch_size - input_dim

        # Patch embedding: (B, 2, patch_size) -> (B, n_patches, embed_dim)
        # We flatten the 2 channels into the patch features: 2 * patch_size
        self.patch_embed = nn.Linear(2 * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent space
        self.to_mu = nn.Linear(embed_dim, z_dim)
        self.to_logvar = nn.Linear(embed_dim, z_dim)

        # Decoder: Latent -> Patches -> (2, patch_size)
        self.latent_to_patches = nn.Linear(z_dim, self.n_patches * embed_dim)
        self.patch_to_pixels = nn.Linear(embed_dim, 2 * patch_size)

        # Classifier
        self.classifier_mlp = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def patchify(self, x_dna, x_gene):
        """
        dna, gene: (B, L)
        Returns: (B, n_patches, 2 * patch_size)
        """
        B = x_dna.shape[0]
        # Stack as 2 channels: (B, 2, L)
        x = torch.stack([x_dna, x_gene], dim=1)
        
        if self.pad_len > 0:
            pad = x.new_zeros((B, 2, self.pad_len))
            x = torch.cat([x, pad], dim=2)
            
        # (B, 2, n_patches * patch_size) -> (B, 2, n_patches, patch_size)
        x = x.view(B, 2, self.n_patches, self.patch_size)
        # Permute to (B, n_patches, 2, patch_size) then flatten channels/pixels
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, 2 * self.patch_size)
        return x

    def unpatchify(self, patches):
        """
        patches: (B, n_patches, 2 * patch_size)
        Returns: (B, 2, input_dim)
        """
        B = patches.shape[0]
        # Reshape to (B, n_patches, 2, patch_size)
        x = patches.view(B, self.n_patches, 2, self.patch_size)
        # Permute to (B, 2, n_patches, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, 2, self.n_patches * self.patch_size)
        # Trim padding
        if x.shape[2] > self.input_dim:
            x = x[:, :, :self.input_dim]
        return x

    def encode(self, x_dna, x_gene):
        patches = self.patchify(x_dna, x_gene)
        emb = self.patch_embed(patches)
        emb = emb + self.pos_embed
        emb = self.dropout(emb)
        
        tokens = self.transformer(emb)
        pooled = tokens.mean(dim=1)
        
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.shape[0]
        patch_emb = self.latent_to_patches(z).view(B, self.n_patches, self.embed_dim)
        pixel_patches = self.patch_to_pixels(patch_emb)
        recon_combined = self.unpatchify(pixel_patches)
        return recon_combined[:, 0, :], recon_combined[:, 1, :]

    def forward_with_classifier(self, x_dna, x_gene):
        mu, logvar = self.encode(x_dna, x_gene)
        z = self.reparameterize(mu, logvar)
        recon_dna, recon_gene = self.decode(z)
        class_logits = self.classifier_mlp(mu)
        return recon_dna, recon_gene, mu, logvar, class_logits

    def loss(
        self,
        x_dna,
        x_gene,
        r_dna,
        r_gene,
        mu,
        logvar,
        labels,
        preds,
        lamb=1.0,
        alpha=20.0,
    ):
        # Using celoss from utils, matching original CNNVAE
        r_l_dna = celoss(x_dna.unsqueeze(1), r_dna.unsqueeze(1))
        r_l_gene = celoss(x_gene.unsqueeze(1), r_gene.unsqueeze(1))
        recon_loss = r_l_dna + r_l_gene

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl)

        # Cross entropy for classification
        cls_loss = F.cross_entropy(preds, labels.view(-1))

        total_loss = recon_loss + (lamb * kl_loss) + (alpha * cls_loss)
        return total_loss, r_l_dna, r_l_gene, kl_loss, cls_loss

    def trainer(
        self,
        train_loader,
        optimizer,
        num_epochs: int,
        device=None,
        lamb: float = 1.0,
        alpha: float = 20.0,
        log_interval: int = 1,
        patience: int = 10,
        verbose: bool = True,
        test_loader=None,
        apply_masking: bool = False,
        mask_ratio: float = 0.2,
        block_size: int = 200,
        warmup_epochs: int = 20
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Try to detect wandb
        wandb_run = None
        try:
            import wandb
            if wandb.run is not None:
                wandb_run = wandb.run
        except ImportError:
            pass

        history = []
        mu_logvar_history = []
        test_history = []

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_total = 0.0
            epoch_recon_m = 0.0
            epoch_recon_g = 0.0
            epoch_kl = 0.0
            epoch_cls = 0.0
            epoch_acc = 0.0
            num_batches = 0

            mu_mean_epoch = 0.0
            mu_std_epoch = 0.0
            logvar_mean_epoch = 0.0
            logvar_std_epoch = 0.0

            if epoch < warmup_epochs:
                current_lamb = (epoch / warmup_epochs) * lamb
            else:
                current_lamb = lamb

            for batch_idx, (x_dna_meth, x_gene_exp, labels) in enumerate(train_loader):
                x_dna_meth = x_dna_meth.to(device)
                x_gene_exp = x_gene_exp.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                recon_dna, recon_gene, mu, logvar, class_logits = self.forward_with_classifier(x_dna_meth, x_gene_exp)

                loss, rec_m, rec_g, kl, cls_loss = self.loss(
                    x_dna=x_dna_meth,
                    x_gene=x_gene_exp,
                    r_dna=recon_dna,
                    r_gene=recon_gene,
                    mu=mu,
                    logvar=logvar,
                    labels=labels,
                    preds=class_logits,
                    lamb=current_lamb,
                    alpha=alpha
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                # Metrics
                preds_cls = torch.argmax(class_logits, dim=1)
                acc = (preds_cls == labels.view(-1)).float().mean().item()

                epoch_total += loss.item()
                epoch_recon_m += rec_m.item()
                epoch_recon_g += rec_g.item()
                epoch_kl += kl.item()
                epoch_cls += cls_loss.item()
                epoch_acc += acc
                num_batches += 1

                mu_mean_epoch += mu.mean().item()
                mu_std_epoch += mu.std().item()
                logvar_mean_epoch += logvar.mean().item()
                logvar_std_epoch += logvar.std().item()

                if verbose and (batch_idx + 1) % log_interval == 0:
                    print(
                        f"mu_norm: {mu.norm(dim=1).mean().item():.3f} logv_norm: {logvar.norm(dim=1).mean().item():.3f} | "
                        f"Acc: {acc:.4f} ClsLoss: {cls_loss.item():.4f}"
                    )
                
                if wandb_run:
                    wandb_run.log({
                        "batch/loss": loss.item(),
                        "batch/acc": acc,
                        "batch/cls_loss": cls_loss.item(),
                        "batch/kl_loss": kl.item(),
                        "mu_norm": mu.norm(dim=1).mean().item(),
                        "logv_norm": logvar.norm(dim=1).mean().item()
                    })

            denom = max(num_batches, 1)
            avg_total = epoch_total / denom
            avg_rec_m = epoch_recon_m / denom
            avg_rec_g = epoch_recon_g / denom
            avg_kl = epoch_kl / denom
            avg_cls = epoch_cls / denom
            avg_acc = epoch_acc / denom

            history.append([avg_total, avg_rec_m, avg_rec_g, avg_kl, avg_cls, avg_acc])
            mu_logvar_history.append([
                (mu_mean_epoch / denom),
                (mu_std_epoch / denom),
                (logvar_mean_epoch / denom),
                (logvar_std_epoch / denom),
            ])

            # Validation
            if test_loader is not None:
                self.eval()
                t_total = t_rec_m = t_rec_g = t_kl = t_cls = t_acc = 0.0
                t_batches = 0
                with torch.no_grad():
                    for x_dna_meth, x_gene_exp, labels in test_loader:
                        x_dna_meth, x_gene_exp, labels = x_dna_meth.to(device), x_gene_exp.to(device), labels.to(device)
                        recon_dna, recon_gene, mu, logvar, class_logits = self.forward_with_classifier(x_dna_meth, x_gene_exp)
                        v_loss, v_rec_m, v_rec_g, v_kl, v_cls_loss = self.loss(
                            x_dna_meth, x_gene_exp, recon_dna, recon_gene, mu, logvar, labels, class_logits, current_lamb, alpha
                        )
                        t_total += v_loss.item()
                        t_rec_m += v_rec_m.item()
                        t_rec_g += v_rec_g.item()
                        t_kl += v_kl.item()
                        t_cls += v_cls_loss.item()
                        preds_cls = torch.argmax(class_logits, dim=1)
                        t_acc += (preds_cls == labels.view(-1)).float().mean().item()
                        t_batches += 1
                
                d = max(t_batches, 1)
                val_metrics = [t_total / d, t_rec_m / d, t_rec_g / d, t_kl / d, t_cls / d, t_acc / d]
                test_history.append(val_metrics)

            if verbose:
                msg = (f"Epoch [{epoch}/{num_epochs}] Train: Tot {avg_total:.2f} | ReconM {avg_rec_m:.4f} | "
                       f"ReconG {avg_rec_g:.4f} | KL {avg_kl:.2f} | Cls {avg_cls:.3f} (Acc {avg_acc:.3f})")
                if test_loader is not None:
                    tt = test_history[-1]
                    msg += f"\n             Test : Tot {tt[0]:.2f} | KL {tt[3]:.2f} | Cls {tt[4]:.3f} (Acc {tt[5]:.3f})"
                print(msg)

            if wandb_run:
                wandb_run.log({
                    "epoch": epoch,
                    "train/total_loss": avg_total,
                    "train/acc": avg_acc,
                    "val/total_loss": test_history[-1][0] if test_loader else None,
                    "val/acc": test_history[-1][5] if test_loader else None,
                })

            if best_loss - avg_total > 0.0:
                best_loss = avg_total
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose: print(f"Early stopping triggered after {epoch} epochs.")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        
        return history, mu_logvar_history, test_history

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
