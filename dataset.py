import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.transforms import functional as TF

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None) : 
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self) : 
            return len(self.images)
        
    def __getitem__(self, index) : 
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = np.array (Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)

        if self.transform is not None : 
            augmentation = self.transform(image = image, mask = mask)
            image = augmentation["image"]
            mask = augmentation ["mask"]
        
        return image, mask
class seg_data(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image = TF.to_tensor(image)
        

        return image
    
    