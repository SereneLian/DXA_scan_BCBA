# %%
import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import crop, pad
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import shap
# augment = augment_lib.TrivialAugment()
device = "cuda:3" if torch.cuda.is_available() else "cpu"



def crop_hf(input_image):
    input_image = np.array(input_image)
    input_image = backgroup_map(input_image)
    input_image = Image.fromarray(input_image)
    hights = np.shape(input_image)[0]
    width = np.shape(input_image)[1]
    upper = int(hights/6)
    image = crop(input_image, upper,0, width-1, width-1)
    return image

def pad_bd(cropped_img):
    target_size = [max(np.shape(cropped_img)), max(np.shape(cropped_img))]
    pad_height = max(0, (target_size[0] - np.shape(cropped_img)[0]) // 2)
    pad_width = max(0, (target_size[1] - np.shape(cropped_img)[1]) // 2)
    # Pad the array
    padded_arr = pad(cropped_img, (pad_width, pad_height, pad_width, pad_height))
    return padded_arr

def backgroup_map(image):
    # if mean_col < 100:
    image[np.where(image >= [250])] = [0]
    return image



train_preprocess = transforms.Compose([
    transforms.Lambda(crop_hf),
    # transforms.Lambda(pad_bd),
    transforms.Resize(224),
    augment,
    transforms.ToTensor(), # PyTorch automatically converts all images into [0,1]. 
    transforms.Normalize([0.5], [0.25])
])

preprocess = transforms.Compose([
    transforms.Lambda(crop_hf),
    # transforms.Lambda(pad_bd),
    transforms.Resize(224),
    transforms.ToTensor(), # PyTorch automatically converts all images into [0,1]. 
    transforms.Normalize([0.5], [0.25])
])

class DXADataset(Dataset):
    def __init__(self, csv_path, transform=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_path1 = self.df['path_11']
        img_path2 = self.df['path_12']
        labels = torch.from_numpy(np.array(self.df['Age when attended assessment centre | Instance 2'])).float()
        label = labels[idx]
        image_filepath1 = img_path1[idx]
        image1 = Image.open(image_filepath1)
        image1 = self.transform(image1)

        image_filepath2 = img_path2[idx]
        image2 = Image.open(image_filepath2)
        image2 = self.transform(image2)
        # print(image2.size())
        image =  torch.concat((image1,image2))
        # print(image.size())
        # return image1, label
        return image, label


# %%
    
targets ='All'
all_dataset = DXADataset('all_patients.csv',preprocess)    

train_dataset = DXADataset('train.csv',preprocess)
valid_dataset = DXADataset('val.csv',preprocess) 
test_dataset = DXADataset('test.csv',preprocess)

pth = '' # load the model path

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False)

class DenseNetRegression(nn.Module):
    def __init__(self):
        super(DenseNetRegression, self).__init__()
        self.densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet.features.conv0 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Adjust other layers if needed
        self.fc = nn.Linear(1024, 1)  # Output layer with 1 neuron for regression task

    def forward(self, x):
        features = self.densenet.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        return out


# %%
model = DenseNetRegression()
model.load_state_dict(torch.load(pth))
model = model.to(device)
model.eval()

# %%
criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.L1Loss()
with torch.no_grad():
    y_train_true = []
    y_train_pred = []
    train_loss1 = []
    train_loss2 = []

    for _, (inputs, label) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        label = label.unsqueeze(1).to(device)
        outputs = model(inputs)
        loss1 = criterion1(outputs, label)
        loss2 = criterion2(outputs, label)
        train_loss1.append(loss1.item())
        train_loss2.append(loss2.item())
        y_train_true.extend(label.cpu().detach().numpy())
        y_train_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
    print('Train MSE loss:', np.average(train_loss1))
    print('Train MAE loss:', np.average(train_loss2))
    print('Train MAPE loss:', mean_absolute_percentage_error(y_train_true, y_train_pred))
    print('Train R2 Score:', r2_score(y_train_true, y_train_pred))

with torch.no_grad():
    y_val_true = []
    y_val_pred = []
    val_loss1 = []
    val_loss2 = []
    for _, (inputs, label) in enumerate(tqdm(valid_loader)):
        inputs = inputs.to(device)
        label = label.unsqueeze(1).to(device)
        outputs = model(inputs)
        loss1 = criterion1(outputs, label)
        loss2 = criterion2(outputs, label)
        val_loss1.append(loss1.item())
        val_loss2.append(loss2.item())
        y_val_true.extend(label.cpu().detach().numpy())
        y_val_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
    print('Val MSE loss:', np.average(val_loss1))
    print('Val MAE loss:', np.average(val_loss2))
    print('Val MAPE loss:', mean_absolute_percentage_error(y_val_true, y_val_pred))
    print('Val R2 Score:', r2_score(y_val_true, y_val_pred))

with torch.no_grad():
    y_test_true = []
    y_test_pred = []
    test_loss1 = []
    test_loss2 = []
    for _, (inputs, label) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        label = label.unsqueeze(1).to(device)
        outputs = model(inputs)
        loss1 = criterion1(outputs, label)
        loss2 = criterion2(outputs, label)
        test_loss1.append(loss1.item())
        test_loss2.append(loss2.item())
        y_test_true.extend(label.cpu().detach().numpy())
        y_test_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
    print('Test MSE loss:', np.average(test_loss1))
    print('Test MAE loss:', np.average(test_loss2))
    print('Test MAPE loss:', mean_absolute_percentage_error(y_test_true, y_test_pred))
    print('Test R2 Score:', r2_score(y_test_true, y_test_pred))

# %%
with torch.no_grad():
    y_t2dm_true = []
    y_t2dm_pred = []
    for _, (inputs, label) in enumerate(tqdm(all_loader)):
        inputs = inputs.to(device)
        label = label.unsqueeze(1).to(device)
        outputs = model(inputs)
        y_t2dm_true.extend(label.cpu().detach().numpy())
        y_t2dm_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())

# %%

all_df = pd.read_csv('new_data/all_patients.csv')
all_df['B-Age'] = y_t2dm_pred
all_df.to_csv('new_results/'+ pth.split('/')[-2]+'_pred.csv', index=False)
