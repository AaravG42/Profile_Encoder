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
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            mask = torch.rand_like(x) > self.mask_ratio
            return x * mask
        return x


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


def get_genomic_train_transforms():
    """Get training transforms for genomic self-supervised learning."""
    return transforms.Compose(
        [
            GaussianNoise(std=0.1, prob=0.8),
            RandomMasking(mask_ratio=0.15, prob=0.5),
            ChannelDropout(prob=0.1),
        ]
    )


def get_genomic_val_transforms():
    """Get validation transforms for genomic data (no augmentation)."""
    return transforms.Compose([])


class GenomicDataset(torch.utils.data.Dataset):
    """
    Dataset for genomic data (gene expression + DNA methylation).
    
    Loads 1D tensors of shape (15703,) and keeps them as 1D vectors
    of shape (2, 15703) where:
    - Channel 0: Gene expression
    - Channel 1: DNA methylation
    """

    def __init__(
        self,
        methylation_path,
        gene_expression_path,
        labels_path,
        transform=None,
        num_crops=2,
    ):
        """
        Args:
            methylation_path: Path to methylation pickle file
            gene_expression_path: Path to gene expression pickle file
            labels_path: Path to labels pickle file
            transform: Optional transform to be applied on a sample
            num_crops: Number of augmented views to create
        """
        import pickle

        # Load data
        with open(methylation_path, "rb") as f:
            self.methylation = pickle.load(f)
        
        with open(gene_expression_path, "rb") as f:
            self.gene_expression = pickle.load(f)
        
        try:
            with open(labels_path, "rb") as f:
                self.labels = pickle.load(f)
            
            # Handle one-hot encoded labels
            if self.labels.dim() > 1 and self.labels.shape[1] > 1:
                self.labels = torch.argmax(self.labels, dim=1)
        except FileNotFoundError:
            print(f"Labels file not found at {labels_path}, creating dummy labels.")
            self.labels = torch.zeros(len(self.methylation), dtype=torch.long)
        
        # Convert to float32 if needed
        self.methylation = self.methylation.to(dtype=torch.float32)
        self.gene_expression = self.gene_expression.to(dtype=torch.float32)
        self.labels = self.labels.to(dtype=torch.long)
        
        self.transform = transform
        self.num_crops = num_crops
        
        # Validate data dimensions
        assert len(self.methylation) == len(self.gene_expression) == len(self.labels), \
            "Data length mismatch"
        assert self.methylation.shape[1] == 15703, \
            f"Expected 15703 features, got {self.methylation.shape[1]}"
        
        print(f"Loaded genomic dataset: {len(self)} samples")
        print(f"  Methylation shape: {self.methylation.shape}")
        print(f"  Gene expression shape: {self.gene_expression.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Number of classes: {len(torch.unique(self.labels))}")

    def __len__(self):
        return len(self.methylation)

    def __getitem__(self, idx):
        """
        Returns:
            views: List of augmented views (each view is a 2-channel 1D vector of shape (2, 15703))
            label: Class label
        """
        # Get 1D data
        meth = self.methylation[idx]  # Shape: (15703,)
        gene = self.gene_expression[idx]  # Shape: (15703,)
        label = self.labels[idx]
        
        # Stack into 2-channel 1D vector: (2, 15703)
        vector = torch.stack([gene, meth], dim=0)
        
        # Apply augmentations to create multiple views
        if self.transform is not None:
            views = [self.transform(vector.clone()) for _ in range(self.num_crops)]
        else:
            views = [vector for _ in range(self.num_crops)]
        
        return views, label
