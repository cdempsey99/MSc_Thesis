import rasterio
import bisect
from rasterio.windows import Window
from configs.config import *
from torch.utils.data import Dataset, DataLoader
from utils.misc import *

# Fn to ingest FBP images, taking the image found at 'path'
def ingest_fbp_patch(path, x_offset=1000, y_offset=1000):
    with rasterio.open(path) as src:
        # Open a 224 x 224 window so we don't overload RAM
        win = Window(x_offset, y_offset, 224, 224)

        # Read the four bands, normally in order NIR R G B
        img = src.read(window=win)

        # Convert to float and normalise
        # (it will be currently 0-255 but Clay expects 0-1)
        img_tensor = torch.from_numpy(img).float() / 255.0

        # Normalise?
        mean = torch.tensor([0.485, 0.456, 0.406, 0.406]).view(4, 1, 1)  # Adjust for 4 bands
        std = torch.tensor([0.229, 0.224, 0.225, 0.225]).view(4, 1, 1)
        img_tensor = (img_tensor - mean) / std

    # Add a batch dim, so [1, 4, 224, 224]
    return img_tensor.unsqueeze(0)


# x and y here are the offsets to give us the tile location
def ingest_paired_patch(img_path, mask_path, x, y):
    # 1. Grab the Image Patch
    with rasterio.open(img_path) as src:
        win = Window(x, y, 224, 224)
        img = src.read(window=win)
        # ... (your existing normalization code) ...
        img_tensor = torch.from_numpy(img).float() / 255.0

    # 2. Grab the Mask Patch from the EXACT SAME window
    # Note: We use PIL or Rasterio here; I'll stick to Rasterio for consistency
    with rasterio.open(mask_path) as src_mask:
        mask = src_mask.read(1, window=win)  # Read only the 1st band (the labels)
        mask_tensor = torch.from_numpy(mask).long()  # Labels must be Long integers

    return img_tensor.unsqueeze(0), mask_tensor.unsqueeze(0)


class FBPPatchDataset(Dataset):

    def __init__(self, img_paths, mask_paths, patch_size=224, stride=112, preload=True, max_samples=None):
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


# update the class to read presaved .pt files of the encoder representation of the image tiles
# update to load all packed .pt files into RAM
class BakedFeatureDatasetOld(Dataset):
    def __init__(self, split_dir):
        # List all packed files
        self.files = sorted(list(Path(split_dir).glob("*_packed.pt")))
        self.cumulative_sizes = []
        self.current_total = 0

        # We need to know how many patches are in each file to index correctly
        for f in self.files:
            # We load just the metadata/shape to be fast
            temp_data = torch.load(f, map_location='cpu', weights_only=True)
            num_patches = temp_data['features'].size(0)
            self.current_total += num_patches
            self.cumulative_sizes.append(self.current_total)
            # Make sure to delete each of these at the end of the loop to minimise RAM
            del temp_data

        # Keep track of which file is currently "open" in RAM to avoid constant reloading
        self.active_file_idx = -1
        self.active_features = None
        self.active_masks = None

    def __len__(self):
        return self.current_total

    def __getitem__(self, idx):
        # 1. Find which file contains this global index
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        # 2. If it's not the file we already have in RAM, load it
        if file_idx != self.active_file_idx:
            data = torch.load(self.files[file_idx], map_location='cpu')
            self.active_features = data['features']
            self.active_masks = data['masks']
            self.active_file_idx = file_idx

        # 3. Calculate local index within that file
        offset = self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - offset

        return self.active_features[local_idx], self.active_masks[local_idx]


class BakedFeatureDatasetOlder(Dataset):
    def __init__(self, file_paths):
        self.file_paths = sorted(file_paths)
        self.cumulative_sizes = []
        total_count = 0

        # Pre-calculate the "map" of where each file starts and ends
        for p in self.file_paths:
            # We load just the metadata/shape to avoid RAM bloat
            data = torch.load(p, map_location="cpu")
            num_patches = data['features'].size(0)
            total_count += num_patches
            self.cumulative_sizes.append(total_count)
            del data  # Clear immediately

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        # 1. Find which file the 'idx' belongs to using binary search
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        # 2. Calculate the local index within that file
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        # 3. Load the file (In your training script, num_workers=0 makes this stable)
        # For high-scale, you might want to cache the 'active' file to avoid re-loading
        path = self.file_paths[file_idx]
        data = torch.load(path, map_location="cpu")

        feature = data['features'][local_idx]
        mask = data['masks'][local_idx]

        return feature, mask

    import bisect
    import torch
    from torch.utils.data import Dataset

class BakedFeatureDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = sorted(file_paths)
        self.cumulative_sizes = []
        total_count = 0

        # Persistent Buffer: This is the robust part.
        # It stays empty until the first patch is requested.
        self._current_file_path = None
        self._current_data = None

        for p in self.file_paths:
            # We only load to get the count, then discard.
            # map_location="cpu" is vital to prevent GPU spikes.
            data = torch.load(p, map_location="cpu")
            num_patches = data['features'].size(0)
            total_count += num_patches
            self.cumulative_sizes.append(total_count)
            del data

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        path = self.file_paths[file_idx]

        # SMART BUFFER:
        # If the requested patch is in the file we already have in RAM,
        # we skip the disk I/O entirely.
        if path != self._current_file_path:
            self._current_file_path = path
            # Load the new file into the buffer
            self._current_data = torch.load(path, map_location="cpu")

        # Pull tensors from the buffer
        feature = self._current_data['features'][local_idx]
        mask = self._current_data['masks'][local_idx]

        return feature, mask
