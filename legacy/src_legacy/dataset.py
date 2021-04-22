import os
from PIL import Image
import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


# Define train datset & test datset class
class TrainDataset(Dataset):
    def __init__(self, img_paths, csv, transform):
        self.img_paths = img_paths
        self.csv = csv
        self.transform = transform

    def __getitem__(self, index):
        # get persion id and mask info
        pid, mask_id = divmod(index, 7)
        
        # get image
        mask_images = [d for d in os.listdir(self.img_paths[pid]) if not d.startswith('._')]
        image = Image.open(os.path.join(self.img_paths[pid], mask_images[mask_id]))


        if self.transform:
            image = self.transform(image)
            
        #get label info
        gender = 0 if self.csv.gender[pid] == 'male' else 1
        age = 0 if self.csv.age[pid] < 30 else 1 if 30 <= self.csv.age[pid] < 60 else 2
        mask = 1 if 'incorrect' in mask_images[mask_id] else 2 if 'normal' in mask_images[mask_id] else 0
        
        label = 6 * mask + 3 * gender + age
        
        return image, label

    def __len__(self):
        return len(self.img_paths) * 7
    
    
class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
    
    
def get_train_dataloader(train_dataset, batch_size, num_workers):
    
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

def get_valid_dataloader(valid_dataset, batch_size, num_workers):    
    return DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

def get_test_dataloader(test_dataset):    
    return DataLoader(
        test_dataset,
        shuffle=False
    )