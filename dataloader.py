import os
import torch
import glob
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F 
from torch.utils.data import Dataset
from PIL import Image
    
class DistrictDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        datalength = len(self.metadata)
        folder_idx = self.metadata.iloc[idx, 0]
       
        image_root_path = "{}{}".format(self.root_dir, folder_idx)
        images = np.stack([io.imread("{}/{}".format(image_root_path, x)) / 255.0 for x in os.listdir(image_root_path)])    
        sample = {'images': images, 'directory': folder_idx, 'num': len(images)}
        
        if self.transform:
            sample['images'] = self.transform(sample['images'])

        return sample  
    
    
class ReducedDataset(Dataset):
    def __init__(self, metadata, root_dir):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        dir_list = self.metadata['Directory'].values.tolist()
        if idx not in dir_list: 
            print(idx)
            return -1
        feature_matrix = np.genfromtxt("{}{}.csv".format(self.root_dir, idx), delimiter=' ')                              
        sample = {'images': feature_matrix, 'directory': idx, 'num': len(feature_matrix)}
        return sample  


class EmbeddingDataset(Dataset):
    def __init__(self, metadata,  root_dir, transform = None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 0] + '.png'
        district = int(self.metadata.iloc[idx, 1])
        image_root_path = "{}{}".format(self.root_dir, district)
        image = Image.open("{}/{}".format(image_root_path, img_name))
                
        if self.transform:
            image = self.transform(image)
            
        return image, idx
    
class NproxyDataset(Dataset):
    def __init__(self, metadata, root_dir, transform = None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_name = self.metadata.iloc[idx]['y_x']
        nl = float(self.metadata.iloc[idx]['light_sum']  / 16.0)
        
        image_path = "{}{}.png".format(self.root_dir, image_name)
        image = io.imread(image_path) / 255.0
        
        if self.transform:
            image = self.transform(np.stack([image])).squeeze()
            
        return image, nl
    
class OproxyDataset(Dataset):
    def __init__(self, metadata, root_dir, transform = None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_id, y_urban, y_rural, y_env = self.metadata.iloc[idx, :].values
        image_path = "{}{}.png".format(self.root_dir, int(image_id))
        image = io.imread(image_path) / 255.0
        
        if self.transform:
            image = self.transform(np.stack([image])).squeeze()
            
        return image, torch.Tensor([y_urban, y_rural, y_env])   
    
class RandomRotate(object):
    def __call__(self, images):
        rotated = np.stack([self.random_rotate(x) for x in images])
        return rotated
    
    def random_rotate(self, image):
        rand_num = np.random.randint(0, 4)
        if rand_num == 0:
            return np.rot90(image, k=1, axes=(0, 1))
        elif rand_num == 1:
            return np.rot90(image, k=2, axes=(0, 1))
        elif rand_num == 2:
            return np.rot90(image, k=3, axes=(0, 1))   
        else:
            return image

class Grayscale(object):
    def __init__(self, prob = 1):
        self.prob = prob

    def __call__(self, images):     
        random_num = np.random.randint(100, size=1)[0]
        if random_num <= self.prob * 100:
            gray_images = (images[:, 0, :, :] + images[:, 1, :, :] + images[:, 2, :, :]) / 3
            gray_scaled = gray_images.unsqueeze(1).repeat(1, 3, 1, 1)
            return gray_scaled
        else:
            return images        
        
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images]) 
        return normalized
        
class ToTensor(object):
    def __call__(self, images):
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float() 