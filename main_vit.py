import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score
import augment_lib
import timm
from model.dataset import DXA_CNN_Dataset
from utils.utils import EarlyStopping, cnn_center_crop, make_time_folder

learning_rate = 0.00001
patience = 15
epochs = 300
weight_decay = 0.00001
augment = augment_lib.TrivialAugment()


def sample_weights(tran_data):
    tran_data['AL'] = pd.cut(x=tran_data['Age when attended assessment centre | Instance 2'], bins=[40, 50, 60, 70, 90], labels=[0, 1, 2, 3])
    class_counts = [tran_data['AL'].value_counts()[0],  tran_data['AL'].value_counts()[1],
            tran_data['AL'].value_counts()[2],tran_data['AL'].value_counts()[3]]
    labels =  tran_data['AL'].to_list()
    num_samples = len(tran_data)
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(num_samples)]

    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)
    return sampler


train_preprocess = transforms.Compose([
    transforms.Lambda(cnn_center_crop),
    augment,
    transforms.ToTensor(), # PyTorch automatically converts all images into [0,1]. 
    transforms.Resize(224),
    transforms.Normalize([0.5], [0.5])
])
preprocess = transforms.Compose([
    transforms.Lambda(cnn_center_crop),
    transforms.ToTensor(), # PyTorch automatically converts all images into [0,1]. 
    transforms.Resize(224),
    transforms.Normalize([0.5], [0.5])
])


train_dataset = DXA_CNN_Dataset('HC_train.csv',train_preprocess)
valid_dataset = DXA_CNN_Dataset('HC_val.csv',preprocess) 
test_dataset = DXA_CNN_Dataset('HC_test.csv',preprocess)
pre_dataset = DXA_CNN_Dataset('pre_dis.csv',preprocess)  
post_dataset = DXA_CNN_Dataset('post_dis.csv',preprocess)    

# train_data = pd.read_csv('new_processed_data/HC_train.csv')
# train_sampler = sample_weights(train_data)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, sampler=train_sampler)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
pre_loader = DataLoader(pre_dataset, batch_size=64, shuffle=False)
post_loader = DataLoader(post_dataset, batch_size=64, shuffle=False)

model = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=2, num_classes=1)
model = model.cuda()


optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay = weight_decay)
scheduler =torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)

def train(model, model_save_folder, optimizer= optimizer, scheduler=scheduler):
    best_r2 = 0
    model_save_path = 'new_results/'+ model_save_folder+make_time_folder() +'/'
    os.makedirs(model_save_path)

    criterion = torch.nn.MSELoss()
    train_loss = []
    valid_loss = []
    test_loss = []
    pre_loss = []
    post_loss = []
    train_r2 = []
    val_r2 = []
    test_r2 = []
    pre_r2 = []
    post_r2 = []
    train_CBA = []
    val_BCA = []
    test_BCA = []
    pre_BCA = []
    post_BCA = []
    early_stopping = EarlyStopping(patience=patience,verbose=True)

    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
        y_true = []
        y_pred = []
        # =========================train=======================
        for _, (inputs, label) in enumerate(tqdm(train_loader)):
            inputs = inputs.cuda()
            label = label.unsqueeze(1).cuda()
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
        train_ba_ca = [y_pred[i] - y_true[i] for i in range(len(y_true))]
        train_CBA.append(np.mean(train_ba_ca))    
        train_r2.append(train_epoch_r2)
        print("train r2 = {:.3f}, loss = {}".format(train_epoch_r2, np.average(train_epoch_loss)))
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            y_true = []
            y_pred = []
            for _, (inputs, label) in enumerate(tqdm(valid_loader)):
                inputs = inputs.cuda() 
                label = label.unsqueeze(1).cuda()
                outputs = model(inputs)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())
                y_true.extend(label.cpu().detach().numpy())
                y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
            valid_loss.append(np.average(val_epoch_loss))
            val_epoch_r2 = r2_score(y_true, y_pred)
            val_r2.append(val_epoch_r2)
            val_ba_ca = [y_pred[i] - y_true[i] for i in range(len(y_true))]
            val_BCA.append(np.mean(val_ba_ca))    
            print("epoch = {}, valid r2 = {:.3f}, loss = {}".format(epoch, val_epoch_r2, np.average(val_epoch_loss)))
            
            test_epoch_loss = []
            y_true = []
            y_pred = []
            for _, (inputs, label) in enumerate(tqdm(test_loader)):
                inputs = inputs.cuda()
                label = label.unsqueeze(1).cuda()
                outputs = model(inputs)
                loss = criterion(outputs, label)
                test_epoch_loss.append(loss.item())
                y_true.extend(label.cpu().detach().numpy())
                y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
            test_epoch_r2 = r2_score(y_true, y_pred)
            test_r2.append(test_epoch_r2)
            test_loss.append(np.average(test_epoch_loss))
            test_ba_ca = [y_pred[i] - y_true[i] for i in range(len(y_true))]
            test_BCA.append(np.mean(test_ba_ca))    
            print("epoch = {}, test r2 = {:.3f}, loss = {}".format(epoch, test_epoch_r2, np.average(test_epoch_loss)))

            if epoch >10 and best_r2 < val_epoch_r2:
                best_r2 = val_epoch_r2
                pre_epoch_loss = []
                y_true = []
                y_pred = []
                for _, (inputs, label) in enumerate(tqdm(pre_loader)):
                    inputs = inputs.cuda()
                    # label = label.to(device)
                    label = label.unsqueeze(1).cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, label)
                    pre_epoch_loss.append(loss.item())
                    y_true.extend(label.cpu().detach().numpy())
                    y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
                pre_epoch_r2 = r2_score(y_true, y_pred)
                pre_r2.append(pre_epoch_r2)
                pre_loss.append(np.average(pre_epoch_loss))
                pre_ba_ca = [y_pred[i] - y_true[i] for i in range(len(y_true))]
                pre_BCA.append(np.mean(pre_ba_ca))    

                print("epoch = {}, pre r2 = {:.3f}, loss = {}".format(epoch, pre_epoch_r2, np.average(pre_loss)))
                
                post_epoch_loss = []
                y_pred = []
                y_true = []
                for _, (inputs, label) in enumerate(tqdm(post_loader)):
                    inputs = inputs.cuda() 
                    # label = label.to(device)
                    label = label.unsqueeze(1).cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, label)
                    post_epoch_loss.append(loss.item())
                    y_true.extend(label.cpu().detach().numpy())
                    y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
                post_epoch_r2 = r2_score(y_true, y_pred)
                post_r2.append(post_epoch_r2)
                post_loss.append(np.average(post_epoch_loss))
                post_ba_ca = [y_pred[i] - y_true[i] for i in range(len(y_true))]
                post_BCA.append(np.mean(post_ba_ca))    
                print("epoch = {}, post r2 = {:.3f}, loss = {}".format(epoch, post_epoch_r2, np.average(post_loss)))
            
            else:
                pre_r2.append(0)
                pre_loss.append(0)
                pre_BCA.append(0)
                post_r2.append(0)
                post_loss.append(0)
                post_BCA.append(0)
            #==================early stopping======================
            early_stopping(valid_loss[-1],model=model,path=model_save_path+'/model_epoch_'+str(epoch))
            if early_stopping.early_stop:
                print("Early stopping")
                break

    df = pd.DataFrame(columns = ['train loss', 'train r2', 'val loss', 'val r2', 'test loss', 'test r2'])
    df['train loss'] = train_loss
    df['val loss'] = valid_loss
    df['test loss'] = test_loss
    df['pre loss'] = pre_loss
    df['post loss'] = post_loss
    df['train r2'] = train_r2
    df['val r2'] = val_r2
    df['test r2'] = test_r2
    df['pre r2'] = pre_r2
    df['post r2'] = post_r2
    df['train BA-CA'] = train_CBA
    df['val BA-CA'] = val_BCA
    df['test BA-CA'] = test_BCA
    df['pre BA-CA'] = pre_BCA
    df['post BA-CA'] = post_BCA
    df.to_csv(model_save_path+'record.csv', index=False)


train(model =model, model_save_folder = 'ViT/')



