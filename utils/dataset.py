import rasterio
import bisect
import random
from rasterio.windows import Window
from configs.config import *
from torch.utils.data import Dataset, DataLoader, Sampler
from utils.misc import *
import json
import numpy as np

# Fn to ingest FBP images, taking the image found at 'path'
def ingest_fbp_patch(path, x_offset=1000, y_offset=1000):
    with rasterio.open(path) as src:
        win = Window(x_offset, y_offset, 224, 224)
        img = src.read([1, 2, 3], window=win).astype(np.float32)

        mean = np.array([80.53, 118.39, 115.69]).reshape(3, 1, 1)
        std = np.array([58.25, 65.24, 63.92]).reshape(3, 1, 1)
        img = (img - mean) / std

        img_tensor = torch.from_numpy(img).float()

    return img_tensor.unsqueeze(0)

def ingest_paired_patch(img_path, mask_path, x, y):
    with rasterio.open(img_path) as src:
        win = Window(x, y, 224, 224)
        img = src.read([1, 2, 3], window=win).astype(np.float32)  # Only 3 real bands

        # Normalise with FBP-specific stats
        mean = np.array([80.53, 118.39, 115.69]).reshape(3, 1, 1)
        std = np.array([58.25, 65.24, 63.92]).reshape(3, 1, 1)
        img = (img - mean) / std

        img_tensor = torch.from_numpy(img).float()

    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1, window=win)
        mask_tensor = torch.from_numpy(mask).long()

    return img_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)


def random_augment(features, mask):
    """
    Applies random spatial augmentations to feature maps and masks.
    Same transform applied to both to maintain alignment.

    Args:
        features: [1024, 28, 28] tensor
        mask: [224, 224] tensor
    Returns:
        augmented features and mask
    """
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        features = torch.flip(features, dims=[2])
        mask = torch.flip(mask, dims=[1])

    # Random vertical flip
    if torch.rand(1) > 0.5:
        features = torch.flip(features, dims=[1])
        mask = torch.flip(mask, dims=[0])

    # Random 90 degree rotation (0, 90, 180, 270)
    k = torch.randint(0, 4, (1,)).item()
    features = torch.rot90(features, k, dims=[1, 2])
    mask = torch.rot90(mask, k, dims=[0, 1])

    return features, mask


class FBPPatchDataset(Dataset):

    def __init__(self, img_paths, mask_paths, patch_size=224, stride=112, preload=False, max_samples=None):
        self.patch_size = patch_size
        self.preload = preload
        self.samples = [] # List of [img_path, mask_path, x, y]
        self.loaded_data = []

        for img_p, mask_p in zip(img_paths, mask_paths):
            if max_samples and len(self.samples) >= max_samples:
                break

            with rasterio.open(img_p) as src:
                h, w = src.height, src.width

                # Adjusting this to remove the min_labelled_pixels check
                # TODO: removed the MIN_LABELLED_PIXELS check for speed reasons. Think about if we should put a similar check back in
                """
                # Create a grid of x, y offsets
                for y in range(0, h - patch_size, stride):
                    if max_samples and len(self.samples) >= max_samples:
                        break

                    for x in range(0, w - patch_size, stride):
                        if max_samples and len(self.samples) >= max_samples:
                            break

                        # Check here if the patch has any pixels that are not label 0
                        win = Window(x, y, patch_size, patch_size)
                        mask_patch = src.read(1, window=win)

                        # Only use if there are less than some minimum number of labelled pixels (500 for now)
                        # Could try bringing this number up to only take interesting pixels?
                        if (mask_patch > 0).sum() > MIN_LABELLED_PIXELS:
                            self.samples.append((img_p, mask_p, x, y))
                """

            # Mathematically generate the grid without reading any pixels
            for y in range(0, h - patch_size, stride):
                for x in range(0, w - patch_size, stride):
                    self.samples.append((img_p, mask_p, x, y))

                    if max_samples and len(self.samples) >= max_samples:
                        break
                if max_samples and len(self.samples) >= max_samples:
                    break

        # 2. Pre-loading with Progress Bar
        if self.preload:
            log_msg(f"Pre-loading {len(self.samples)} patches into HPC RAM...")
            for i, (img_p, mask_p, x, y) in enumerate(self.samples):
                img_t, mask_t = ingest_paired_patch(img_p, mask_p, x, y)
                # Store as CPU tensors to keep GPU memory free for the model
                self.loaded_data.append((img_t.squeeze(0).cpu(), mask_t.squeeze(0).cpu()))

                if i % 500 == 0:
                    log_msg(f"Loaded {i}/{len(self.samples)} patches...")
            log_msg("Pre-loading complete.")

    # Note the __x__ here as __len__ will be called automatically by Python when we use len(dataset)
    def __len__(self):
        return len(self.samples)

    # __x__ here as this means automatically used when we write dataset[some_index]
    def __getitem__(self, idx):
        img_p, mask_p, x, y = self.samples[idx]

        # Use the existing ingestion fn
        img_tensor, mask_tensor = ingest_paired_patch(img_p, mask_p, x, y)

        return img_tensor.squeeze(0), mask_tensor.squeeze(0)



class BakedFeatureDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        self.file_paths = sorted(file_paths)
        self.augment = augment

        # Read patches_per_image from metadata
        meta_path = self.file_paths[0].parent / "metadata.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        patches_per_image = meta["patches_per_image"]

        # Build cumulative sizes without loading any tensors
        self.cumulative_sizes = [
            patches_per_image * (i + 1) for i in range(len(self.file_paths))
        ]

        # Smart buffer
        self._current_file_path = None
        self._current_data = None

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        path = self.file_paths[file_idx]

        if path != self._current_file_path:
            self._current_file_path = path
            self._current_data = torch.load(path, map_location="cpu", mmap=True)

        features = self._current_data['features'][local_idx]
        mask = self._current_data['masks'][local_idx]

        if self.augment:
            features, mask = random_augment(features, mask)

        return features, mask

    def get_patch_info(self, idx):
        """Returns (image_path, local_patch_idx) for a given global index."""

        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        local_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)

        return self.file_paths[file_idx], local_idx


class FileLocalitySampler(Sampler):
    """
    Shuffles at the FILE level (order of files, and patch order within each file) instead
    of shuffling individual patch indices across the whole flattened dataset the way
    DataLoader's default shuffle=True does.

    BakedFeatureDataset caches exactly one file at a time (_current_file_path/_current_data,
    reloaded via torch.load(mmap=True) whenever __getitem__ is asked for a patch from a
    different file). Patch-level shuffling picks indices randomly across every file in the
    dataset, so with more than a handful of files, consecutive __getitem__ calls very likely
    land on different files - forcing a fresh mmap reopen almost every call. Keeping each
    file's patches contiguous in the iteration order lets that cache stay valid for an
    entire file's worth of calls instead, cutting reopens from roughly one per patch to
    roughly one per file, while still reshuffling both the file order and the within-file
    patch order every epoch - real randomization, just at the granularity the caching
    actually works at.

    Written for the per-head bagged loaders (--data_bagging in train_decoders.py), where
    paying the patch-level-shuffle cost independently in M separate loaders (instead of
    once, in one shared loader) turned an already-real inefficiency into a ~100x slowdown
    that dominated everything else in the training step.

    Also merges consecutive occurrences of the SAME underlying file into one combined
    range before shuffling. Bootstrap-with-replacement bagging routinely draws the same
    image more than once, and BakedFeatureDataset sorts its file list on construction, so
    duplicate occurrences are already adjacent in the raw index space - but shuffling each
    occurrence independently (the original version of this sampler) scatters them apart
    anyway, forcing a redundant reopen of a file already cached for no reason. Merging
    first means a file drawn K times is visited in one contiguous run instead of K
    scattered ones, so the existing single-file cache is enough - no LRU/multi-file cache
    needed on top of this.
    """
    def __init__(self, dataset):
        self.cumulative_sizes = dataset.cumulative_sizes
        self.file_paths = dataset.file_paths

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __iter__(self):
        merged_ranges = []
        start = 0
        prev_path = None
        for i, end in enumerate(self.cumulative_sizes):
            path = self.file_paths[i]
            if path == prev_path:
                merged_ranges[-1] = (merged_ranges[-1][0], end)
            else:
                merged_ranges.append((start, end))
            prev_path = path
            start = end

        random.shuffle(merged_ranges)

        for start, end in merged_ranges:
            indices = list(range(start, end))
            random.shuffle(indices)
            yield from indices