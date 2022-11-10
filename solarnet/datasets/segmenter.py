import numpy as np
import torch
from pathlib import Path
import random

from typing import Optional, List, Tuple

from .utils import normalize
from .transforms import no_change, horizontal_flip, vertical_flip, colour_jitter
import imgaug.augmenters as iaa

class SegmenterDataset:
    def __init__(self,
                 processed_folder: Path = Path('data/processed'),
                 normalize: bool = True, transform_images: bool = False,
                 device: torch.device = torch.device('cuda:0' if
                                                     torch.cuda.is_available() else 'cpu'),
                 mask: Optional[List[bool]] = None) -> None:

        self.device = device
        self.normalize = normalize
        self.transform_images = transform_images

        if not self.transform_images:
            #labeled dataset
            solar_folder = processed_folder / 'solar'
            empty_folder = processed_folder / 'empty'

            org_solar_files = list((solar_folder / 'org').glob("*.npy"))
            org_empty_files = list((empty_folder / 'org').glob("*.npy"))
            self.x_files = org_solar_files + org_empty_files


            mask_solar_files = [solar_folder / 'mask' / f.name for f in org_solar_files]
            mask_empty_files = [empty_folder / 'mask' / f.name for f in org_empty_files]
            self.y = mask_solar_files + mask_empty_files
        else:
            #unlabeled dataset
            self.x_files = list((processed_folder).glob('*.npy'))
            self.y = list((processed_folder).glob('*.npy')) #ignored

        if mask is not None:
            self.add_mask(mask)

    def add_mask(self, mask: List[bool]) -> None:
        """Add a mask to the data
        """
        assert len(mask) == len(self.x_files), \
            f"Mask is the wrong size! Expected {len(self.x_files)}, got {len(mask)}"
        self.x_files = [x for include, x in zip(mask, self.x_files) if include]
        self.y = [x for include, x in zip(mask, self.y) if include]

    def __len__(self) -> int:
        return len(self.x_files)

    def _transform_images(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #weak augmentation: rotation + elastic distortion
        seq_w = iaa.Sequential([
            iaa.Affine(rotate=(-20,20)),
            iaa.ElasticTransformation(alpha=1,sigma=0.5)
            ])

        #strong augmentation: modify sharpness, contrast and add Gaussian blur
        seq_s = iaa.Sequential([
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.25)),
            iaa.LinearContrast((0.4, 1.6)),
            iaa.GaussianBlur((0, 1.5))
            ])

        self.weak = seq_w(image=image.transpose(1,2,0))
        self.strong= seq_s(image=self.weak).transpose(2,0,1)
        self.weak = self.weak.transpose(2,0,1)
        return self.weak, self.strong

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        x = np.load(self.x_files[index])
        y = np.load(self.y[index])
        if self.transform_images: 
            x = self._transform_images(x)
            if self.normalize:
                x_np = np.array(x)
                x_np = normalize(x_np)
                x = tuple(x_np)
        elif self.normalize:
            x = normalize(x)
        return x, torch.as_tensor(y.copy(), device=self.device).float()
