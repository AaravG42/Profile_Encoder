"""
Dataset and augmentation utilities for self-supervised learning.
"""

import torch
import torch.utils.data
import torchvision.transforms as transforms


class RandomResizedCrop:
    """Random resized crop augmentation."""

    def __init__(self, size, scale=(0.2, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        return transforms.RandomResizedCrop(self.size, scale=self.scale)(img)


class ColorJitter:
    """Color jitter augmentation."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Grayscale:
    """Grayscale augmentation."""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.Grayscale(num_output_channels=3)(img)
        return img


class Solarization:
    """Solarization augmentation."""

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = transforms.functional.solarize(img, threshold=128)
        return img


class HorizontalFlip:
    """Horizontal flip augmentation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.functional.hflip(img)
        return img


def get_train_transforms():
    """Get training transforms for self-supervised learning."""
    transform = transforms.Compose(
        [
            RandomResizedCrop(32, scale=(0.2, 1.0)),
            ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8
            ),
            Grayscale(prob=0.2),
            Solarization(prob=0.1),
            HorizontalFlip(prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    return transform


def get_val_transforms():
    """Get validation transforms."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset that applies augmentations multiple times to create views."""

    def __init__(self, dataset, transform, num_crops=2):
        self.dataset = dataset
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        views = [self.transform(image) for _ in range(self.num_crops)]
        return views, label


# ============================================================================
# Genomic Data Augmentations and Dataset
# ============================================================================


class GaussianNoise:
    """Add Gaussian noise to genomic data."""

    def __init__(self, std=0.1, prob=0.8):
        self.std = std
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class RandomMasking:
    """Random masking augmentation for genomic data."""

    def __init__(self, mask_ratio=0.15, prob=0.5):
        self.mask_ratio = mask_ratio
        # self.prob = prob

    def __call__(self, x):
        # if torch.rand(1) < self.prob:
        #     mask = torch.rand_like(x) > self.mask_ratio
        #     return x * mask
        mask = torch.rand_like(x) > self.mask_ratio
        return x * mask
        # return x


class ChannelDropout:
    """Randomly drop one channel (either gene expression or methylation)."""

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            # x has shape (C, L) where C=2 (gene expression, methylation)
            channel_to_drop = torch.randint(0, 2, (1,)).item()
            x[channel_to_drop] = 0
        return x


def get_genomic_train_transforms(mask_ratio=0.30):
    """Get training transforms for genomic self-supervised learning."""
    return transforms.Compose(
        [
            GaussianNoise(std=0.1, prob=0.8),
            RandomMasking(mask_ratio=mask_ratio, prob=0.5),
        ]
    )


def get_genomic_val_transforms():
    """Get validation transforms for genomic data (no augmentation)."""
    return transforms.Compose([])


class GenomicDataset(torch.utils.data.Dataset):
    """
    Dataset for genomic data (gene expression + DNA methylation).
    
    Loads preprocessed DataFrames where samples are columns and genes are rows.
    Automatically aligns both datasets by common samples and genes.
    
    Returns samples as tensors of shape (2, N_genes) if both channels used.
    """

    def __init__(
        self,
        methylation_path,
        gene_expression_path,
        labels_path=None,  # Not used anymore as labels are in the DataFrames
        transform=None,
        patch_size=None,
        use_channels="both",
    ):
        """
        Args:
            methylation_path: Path to methylation pickle file
            gene_expression_path: Path to gene expression pickle file
            labels_path: (Legacy) Not used, labels are extracted from DataFrames
            transform: Optional transform to be applied on a sample
            patch_size: Optional patch size to reshape the data into (C, N, p)
            use_channels: Which channels to use ('both', 'gene', or 'meth')
        """
        import pandas as pd
        import numpy as np

        # Load data
        print(f"Loading genomic data from:\n  {gene_expression_path}\n  {methylation_path}")
        df_gene = pd.read_pickle(gene_expression_path)
        df_meth = pd.read_pickle(methylation_path)

        # Set gene names as index (first column)
        df_gene.set_index(df_gene.columns[0], inplace=True)
        df_meth.set_index(df_meth.columns[0], inplace=True)

        # 1. Align Samples (Columns)
        common_samples = df_gene.columns.intersection(df_meth.columns)
        print(f"Aligning {len(common_samples)} common samples...")
        
        # 2. Align Genes (Rows) - exclude 'Cancer_Type' row if it exists
        # In this specific dataset format, row 0 (now index 'Cancer_Type') is the labels
        common_genes = df_gene.index.intersection(df_meth.index).drop('Cancer_Type', errors='ignore')
        print(f"Aligning {len(common_genes)} common genes...")

        # Extract labels from 'Cancer_Type' row
        labels_raw = df_gene.loc['Cancer_Type', common_samples]
        label_map = {name: i for i, name in enumerate(sorted(labels_raw.unique()))}
        self.label_names = sorted(labels_raw.unique())
        self.labels = torch.tensor([label_map[l] for l in labels_raw], dtype=torch.long)

        # Extract data and convert to tensors (Samples, Genes)
        self.gene_expression = torch.tensor(
            df_gene.loc[common_genes, common_samples].values.astype(np.float32).T
        )
        self.methylation = torch.tensor(
            df_meth.loc[common_genes, common_samples].values.astype(np.float32).T
        )
        
        self.transform = transform
        self.patch_size = patch_size
        self.use_channels = use_channels
        
        # Validate data dimensions
        assert self.gene_expression.shape[0] == self.methylation.shape[0] == len(self.labels), \
            "Data length mismatch"
        
        print(f"Loaded genomic dataset: {len(self)} samples")
        print(f"  Genes: {len(common_genes)}")
        print(f"  Channels: {self.use_channels}")
        if self.patch_size:
            print(f"  Patching enabled: size={self.patch_size}")
        print(f"  Number of classes: {len(torch.unique(self.labels))}")

    def __len__(self):
        return self.gene_expression.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            vector: Genomic data (possibly reshaped)
            label: Class label
        """
        # Select channels based on config
        if self.use_channels == "both":
            meth = self.methylation[idx]
            gene = self.gene_expression[idx]
            vector = torch.stack([gene, meth], dim=0)  # (2, 17819)
        elif self.use_channels == "gene":
            vector = self.gene_expression[idx].unsqueeze(0)  # (1, 17819)
        elif self.use_channels == "meth":
            vector = self.methylation[idx].unsqueeze(0)  # (1, 17819)
        else:
            raise ValueError(f"Invalid use_channels: {self.use_channels}")

        label = self.labels[idx]
        
        # Pad and reshape if patch_size is provided
        if self.patch_size is not None:
            c, l = vector.shape
            num_patches = (l + self.patch_size - 1) // self.patch_size
            padded_l = num_patches * self.patch_size
            
            if padded_l > l:
                padding = torch.zeros((c, padded_l - l), dtype=vector.dtype, device=vector.device)
                vector = torch.cat([vector, padding], dim=1)
            
            # Reshape to (C, N, P)
            vector = vector.view(c, num_patches, self.patch_size)
        
        # Apply transform if provided
        if self.transform is not None:
            vector = self.transform(vector)
        
        return vector, label
