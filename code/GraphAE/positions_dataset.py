import os
import sys
import PIL
import torch
import pandas as pd
from skimage import io, transform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.utils import prune_bbox

class PositionsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.csv.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, name+"_scene.jpg")
        image_everything = io.imread(img_name)
        img_name = os.path.join(self.root_dir, name+"_silhouette.jpg")
        image_silhouette = io.imread(img_name)[:,:,np.newaxis]
        image_silhouette = np.repeat(image_silhouette, 3, axis=-1)
        image_silhouette_pruned = io.imread(img_name)
        img = PIL.Image.fromarray((image_silhouette_pruned).astype(np.uint8), mode='L')
        img.save(f'../../train/e2e/silhouettes/{name}_.png')
        image_silhouette_pruned = prune_bbox(image_silhouette_pruned)
        np.set_printoptions(threshold=sys.maxsize)
        img = PIL.Image.fromarray((image_silhouette_pruned).astype(np.uint8), mode='L')
        img.save(f'../../train/e2e/silhouettes/{name}__.png')
        image_silhouette_pruned = torch.from_numpy(image_silhouette_pruned)
        tpad = int(image_silhouette_pruned.shape[0] * (15/100))
        bpad = int(image_silhouette_pruned.shape[0] * (5/100))
        hpad = tpad + bpad + image_silhouette_pruned.shape[0] - image_silhouette_pruned.shape[1]
        assert hpad >= 0
        image_silhouette_pruned = torch.nn.functional.pad(image_silhouette_pruned,
            pad=(hpad//2, hpad//2, tpad, bpad))
        image_silhouette_pruned = image_silhouette_pruned.unsqueeze(0).float() / 255
        if self.transform:
            image_silhouette = self.transform(image_silhouette)
            image_everything = self.transform(image_everything)
        pos = self.csv.iloc[idx, 1:]
        pos = np.array([pos["x"], pos["y"], pos["z"]]).astype(np.float32)
        parts = name.split("_")
        scene_name = "_".join(parts[:2])
        recording_name = "_".join(parts[2:7])
        frame_id = "_".join(parts[7:])
        sample = {'cond_position': {'scene': image_everything, 'silhouette': image_silhouette},
                  'cond_shape': image_silhouette_pruned, 'x': pos, 'idx': idx,
                  'name': name, 'recording_name':recording_name, 'scene_name':scene_name, 'frame_id':frame_id}
        return sample
