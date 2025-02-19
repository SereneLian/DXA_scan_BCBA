import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
from torchvision import transforms
from torchvision.transforms.functional import crop, pad
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score
import utils.augment_lib


device = "cuda:0" if torch.cuda.is_available() else "cpu"
augment = augment_lib.TrivialAugment()

learning_rate = 0.001
patience = 15
epochs = 300
weight_decay = 0.00001
targets = "All"


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


### pytorch sampler

train_dataset = DXADataset('train.csv',train_preprocess)
valid_dataset = DXADataset('val.csv',preprocess) 
test_dataset = DXADataset('test.csv',preprocess)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


def make_time_folder(target):
    x = datetime.datetime.now()
    dateTimeStr = str(x)
    return "DenseNet_"+target+"_"+dateTimeStr[5:7]+'_'+dateTimeStr[8:10]+'_'+dateTimeStr[11:13]+'_'+dateTimeStr[14:16] 

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


class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'.pth')
        self.val_loss_min = val_loss


def train():
    model = DenseNetRegression().to(device)
    model_save_path = 'new_results/models/'+make_time_folder(targets) +'/'
    os.makedirs(model_save_path)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay = weight_decay)
    # scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scheduler =torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)
    train_loss = []
    valid_loss = []
    test_loss = []
    train_r2 = []
    val_r2 = []
    test_r2 = []
    early_stopping = EarlyStopping(patience=patience,verbose=True)
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
        y_true = []
        y_pred = []
        # =========================train=======================
        for _, (inputs, label) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            label = label.unsqueeze(1).to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            y_true.extend(label.cpu().detach().numpy())
            y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
        scheduler.step()
        train_epoch_r2 = r2_score(y_true, y_pred)
        train_loss.append(np.average(train_epoch_loss))

        train_r2.append(train_epoch_r2)
        print("train r2 = {:.3f}, loss = {}".format(train_epoch_r2, np.average(train_epoch_loss)))
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            y_true = []
            y_pred = []
            for _, (inputs, label) in enumerate(tqdm(valid_loader)):
                inputs = inputs.to(device)  
                label = label.unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())
                y_true.extend(label.cpu().detach().numpy())
                y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
            valid_loss.append(np.average(val_epoch_loss))
            val_epoch_r2 = r2_score(y_true, y_pred)
            val_r2.append(val_epoch_r2)
            print("epoch = {}, valid r2 = {:.3f}, loss = {}".format(epoch, val_epoch_r2, np.average(val_epoch_loss)))
            test_epoch_loss = []
            y_true = []
            y_pred = []
            for _, (inputs, label) in enumerate(tqdm(test_loader)):
                inputs = inputs.to(device) 
                # label = label.to(device)
                label = label.unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, label)
                test_epoch_loss.append(loss.item())
                y_true.extend(label.cpu().detach().numpy())
                y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
            test_epoch_r2 = r2_score(y_true, y_pred)
            test_r2.append(test_epoch_r2)
            test_loss.append(np.average(test_epoch_loss))
            print("epoch = {}, test r2 = {:.3f}, loss = {}".format(epoch, test_epoch_r2, np.average(test_epoch_loss)))

            #==================early stopping======================
            early_stopping(valid_loss[-1],model=model,path=model_save_path+'/model_epoch_'+str(epoch))
            if early_stopping.early_stop:
                print("Early stopping")
                break

    df = pd.DataFrame(columns = ['train loss', 'train r2', 'val loss', 'val r2', 'test loss', 'test r2'])
    df['train loss'] = train_loss
    df['val loss'] = valid_loss
    df['test loss'] = test_loss
    df['train r2'] = train_r2
    df['val r2'] = val_r2
    df['test r2'] = test_r2
    df.to_csv(model_save_path+'record.csv', index=False)

train()