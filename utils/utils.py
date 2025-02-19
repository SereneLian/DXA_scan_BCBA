import torch
import numpy as np
from tqdm import tqdm


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
        # torch.save(model.state_dict(), path+'model_checkpoint.pth')
        self.val_loss_min = val_loss


def train(model, train_loader, test_loader, device, model_save_path, learning_rate, patience, epochs=100):
    # model_save_path = 'results/'+make_time_folder() +'/'
    # os.makedirs(model_save_path)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
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
            # print(inputs.size())
            label = label.unsqueeze(1).to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            y_true.extend(label.cpu().detach().numpy())
            y_pred.extend(outputs.squeeze(dim=-1).cpu().detach().numpy())
        train_epoch_r2 = r2_score(y_true, y_pred)
        train_loss.append(np.average(train_epoch_loss))

        train_r2.append(train_epoch_r2)
        print("train r2 = {:.3f}, loss = {}".format(train_epoch_r2, np.average(train_epoch_loss)))
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            test_epoch_loss = []
            y_true = []
            y_pred = []
            for _, (inputs, label) in enumerate(tqdm(test_loader)):
                inputs = inputs.to(device) 
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
            #====================adjust lr========================
            # lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            #         10: 5e-7, 15: 1e-7, 20: 5e-8}
            lr_adjust = {5: 0.0001, 10: 1e-5, 20: 5e-6, 30: 1e-6,
                    40: 5e-7, 50: 1e-7, 60: 5e-8}
            if epoch in lr_adjust.keys():
                lr = lr_adjust[epoch]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                print('Updating learning rate to {}'.format(lr))

    df = pd.DataFrame(columns = ['train loss', 'train r2', 'test loss', 'test r2'])
    df['train loss'] = train_loss
    # df['val loss'] = valid_loss
    df['test loss'] = test_loss
    df['train r2'] = train_r2
    # df['val r2'] = val_r2
    df['test r2'] = test_r2
    return df, model
    # df.to_csv(model_save_path+'record.csv', index=False)