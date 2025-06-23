import os.path as osp
import cv2
import numpy as np
import torch
import torch.utils.data as data

class FileDataset(data.Dataset):
    def __init__(self,
                 root_dir,
                 list_path,
                 mode='train',
                 img_size=(640, 480),
                 augment_fn=None,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.list_path = list_path
        self.mode = mode
        self.img_size = img_size
        self.augment_fn = augment_fn

        with open(list_path, 'r') as f:
            self.pair_infos = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        img_name0, img_name1 = self.pair_infos[idx]
        
        # Load images
        img_path0 = osp.join(self.root_dir, "images", img_name0)
        img_path1 = osp.join(self.root_dir, "images", img_name1)

        img0 = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

        img0 = cv2.resize(img0, self.img_size)
        img1 = cv2.resize(img1, self.img_size)

        image0 = torch.from_numpy(img0).float()[None] / 255.
        image1 = torch.from_numpy(img1).float()[None] / 255.

        data = {
            'image0': image0,
            'image1': image1,
            'depth0': torch.empty(0),
            'depth1': torch.empty(0),
            'intrinsic0': torch.empty(0),
            'intrinsic1': torch.empty(0),
            'T_0to1': torch.empty(0),
            'pair_id': idx,
            'pair_names': (img_name0, img_name1),
            'dataset_name': 'File'
        }

        if self.augment_fn:
            data = self.augment_fn(data)

        return data
