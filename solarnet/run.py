import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, SequentialSampler
from torchvision import transforms

import numpy as np
from pathlib import Path
from tqdm import tqdm

from solarnet.preprocessing import MaskMaker, ImageSplitter
from solarnet.datasets import SegmenterDataset, make_masks
from solarnet.models import Segmenter, train_segmenter
import imgaug.augmenters as iaa


class RunTask:

    @staticmethod
    def make_masks(data_folder='data'):
        """Saves masks for each .tif image in the raw dataset. Masks are saved
        in  <org_folder>_mask/<org_filename>.npy where <org_folder> should be the
        city name, as defined in `data/README.md`.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        """
        mask_maker = MaskMaker(data_folder=Path(data_folder))
        mask_maker.process()

    @staticmethod
    def split_images(data_folder='data', imsize=224, empty_ratio=2):
        """Generates images (and their corresponding masks) of height = width = imsize
        for input into the models.

        Parameters
        ----------
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        imsize: int, default: 224
            The size of the images to be generated
        empty_ratio: int, default: 2
            The ratio of images without solar panels to images with solar panels.
            Because images without solar panels are randomly sampled with limited
            patience, having this number slightly > 1 yields a roughly 1:1 ratio.
        """
        splitter = ImageSplitter(data_folder=Path(data_folder))
        splitter.process(imsize=imsize, empty_ratio=empty_ratio)

    @staticmethod
    def train_segmenter(max_epochs=100, val_size=0.1, test_size=0.1, warmup=2,
                        patience=10, data_folder='data', use_classifier=True,
                        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """Train the segmentation model

        Parameters
        ----------
        max_epochs: int, default: 100
            The maximum number of epochs to train for
        warmup: int, default: 2
            The number of epochs for which only the final layers (not from the ResNet base)
            should be trained
        patience: int, default: 5
            The number of epochs to keep training without an improvement in performance on the
            validation set before early stopping
        val_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the validation set
        test_size: float < 1, default: 0.1
            The ratio of the entire dataset to use for the test set
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
        use_classifier: boolean, default: True
            Whether to use the pretrained classifier (saved in data/models/classifier.model by the
            train_classifier step) as the weights for the downsampling step of the segmentation
            model
        device: torch.device, default: cuda if available, else cpu
            The device to train the models on
        """

        data_folder = Path(data_folder)
        model = Segmenter()
        if device.type != 'cpu': model = model.cuda()

        model_dir = data_folder / 'models'
        if use_classifier:
            classifier_sd = torch.load(model_dir / 'classifier.model')
            model.load_base(classifier_sd)

        labeled_folder = data_folder / 'labeled'
        unlabeled_folder = data_folder / 'unlabeled'

        labeled_dataset = SegmenterDataset(processed_folder=labeled_folder)
        unlabeled_dataset = SegmenterDataset(processed_folder=unlabeled_folder, transform_images=True)

        train_mask_l, val_mask_l, test_mask_l = make_masks(len(labeled_dataset), val_size, test_size)
        train_mask_u, _, _  = make_masks(len(unlabeled_dataset), 0, 0)

        labeled_dataset.add_mask(train_mask_l)
        unlabeled_dataset.add_mask(train_mask_u)

        train_dataloader_l = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
        val_dataloader_l = DataLoader(SegmenterDataset(mask=val_mask_l,
                                                     processed_folder=labeled_folder,
                                                     transform_images=False),
                                    batch_size=32, shuffle=True)
        test_dataloader = DataLoader(SegmenterDataset(mask=test_mask_l,
                                                     processed_folder=labeled_folder,
                                                     transform_images=False),
                                    batch_size=32, shuffle=True)
        train_dataloader_u = DataLoader(unlabeled_dataset, batch_size=32)

        train_segmenter(model, train_dataloader_l, train_dataloader_u, val_dataloader_l, max_epochs=max_epochs,
                        warmup=warmup, patience=patience)

        if not model_dir.exists(): model_dir.mkdir()
        torch.save(model.state_dict(), model_dir / 'segmenter.model')

        print("Generating test results")
        images, preds, true = [], [], []
        with torch.no_grad():
            for test_x, test_y in tqdm(test_dataloader):
                test_x, test_y = test_x.float().to(device), test_y.to(device)
                test_preds = model(test_x)
                images.append(test_x.cpu().numpy())
                preds.append(test_preds.squeeze(1).cpu().numpy())
                true.append(test_y.cpu().numpy())
        
        np.save(model_dir / 'segmenter_images.npy', np.concatenate(images))
        np.save(model_dir / 'segmenter_preds.npy', np.concatenate(preds))
        np.save(model_dir / 'segmenter_true.npy', np.concatenate(true))
        

    