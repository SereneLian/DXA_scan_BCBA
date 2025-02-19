
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset



class DXA_CNN_Dataset(Dataset):
    def __init__(self, csv_path, transform=False, stage =2):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.stage = stage
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path1 = self.df['fat_path']
        img_path2 = self.df['bone_path']
        # labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 2'])).float()
        if self.stage ==2:
            labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 2'])).float()
        elif self.stage ==3:
            labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 3'])).float()
        label = labels[idx]
        image_filepath1 = img_path1[idx]
        image1 = Image.open(image_filepath1)
        image1 = self.transform(image1)
        image_filepath2 = img_path2[idx]
        image2 = Image.open(image_filepath2)
        image2 = self.transform(image2)
        image =  torch.concat((image1,image2))
        return image, label


class DXA_CNN_latediff_Dataset(Dataset):
    def __init__(self, csv_path, transform=False, stage =2):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.stage = stage
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path1 = self.df['fat_path']
        img_path2 = self.df['bone_path']
        if self.stage ==2:
            labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 2'])).float()
        elif self.stage ==3:
            labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 3'])).float()
        label = labels[idx]
        image_filepath1 = img_path1[idx]
        image1 = Image.open(image_filepath1)
        image1 = self.transform(image1)
        image_filepath2 = img_path2[idx]
        image2 = Image.open(image_filepath2)
        image2 = self.transform(image2)
        # image =  torch.concat((image1,image2))
        return image1, image2, label


class DXA_CNN_latediff_Gender_Dataset(Dataset):
    def __init__(self, csv_path, transform=False, stage =2):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.stage = stage
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        
        self.df['Sex'] = self.df['Sex'].apply(lambda x: 1 if x == 'M' else 0)
        img_path1 = self.df['fat_path']
        img_path2 = self.df['bone_path']
    
        if self.stage ==2:
            labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 2'])).float()
        elif self.stage ==3:
            labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 3'])).float()
        genders = torch.from_numpy(np.array(self.df['Sex'])).float()
        label = labels[idx]
        gender = genders[idx]
        image_filepath1 = img_path1[idx]
        image1 = Image.open(image_filepath1)
        image1 = self.transform(image1)
        image_filepath2 = img_path2[idx]
        image2 = Image.open(image_filepath2)
        image2 = self.transform(image2)
        # image =  torch.concat((image1,image2))
        return image1, image2, gender, label