
import torch
import torch.nn as nn

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


class DenseNet_latediff_Regression_gender(nn.Module):
    def __init__(self):
        super(DenseNet_latediff_Regression_gender, self).__init__()
        self.densenet1 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.densenet2 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet1.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet2.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Adjust other layers if needed
        self.fc = nn.Linear(2049, 1)  # Output layer with 1 neuron for regression task

    def forward(self, x1, x2, xs):
        features1 = self.densenet1.features(x1)
        out1 = nn.functional.relu(features1, inplace=True)
        out1 = nn.functional.adaptive_avg_pool2d(out1, (1, 1)).view(features1.size(0), -1)
        features2 = self.densenet2.features(x2)
        out2 = nn.functional.relu(features2, inplace=True)
        out2 = nn.functional.adaptive_avg_pool2d(out2, (1, 1)).view(features2.size(0), -1)
        out = torch.concat((out1, out2, xs), dim=-1)
        out = self.fc(out)
        return out


class DenseNet_latediff_Regression(nn.Module):
    def __init__(self):
        super(DenseNet_latediff_Regression, self).__init__()
        self.densenet1 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.densenet2 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet1.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet2.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Adjust other layers if needed
        self.fc = nn.Linear(2048, 1)  # Output layer with 1 neuron for regression task

    def forward(self, x1, x2):
        features1 = self.densenet1.features(x1)
        out1 = nn.functional.relu(features1, inplace=True)
        out1 = nn.functional.adaptive_avg_pool2d(out1, (1, 1)).view(features1.size(0), -1)
        features2 = self.densenet2.features(x2)
        out2 = nn.functional.relu(features2, inplace=True)
        out2 = nn.functional.adaptive_avg_pool2d(out2, (1, 1)).view(features2.size(0), -1)
        out = torch.concat((out1, out2), dim=-1)
        out = self.fc(out)
        return out


class ResNetRegression(nn.Module):
    def __init__(self):
        super(ResNetRegression, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Adjust other layers if needed
        self.resnet.fc = nn.Linear(2048, 1)  # Output layer with 1 neuron for regression task

    def forward(self, x):
        out = self.resnet(x)
        # out = nn.functional.relu(features, inplace=True)
        return out


class ResNet_latediff_Regression(nn.Module):
    def __init__(self):
        super(ResNet_latediff_Regression, self).__init__()
        self.resnet1 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.resnet1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.resnet2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Adjust other layers if needed
        self.resnet1.fc = nn.Linear(2048, 1)  # Output layer with 1 neuron for regression task
        self.resnet2.fc = nn.Linear(2048, 1)  # Output layer with 1 neuron for regression task
        self.fc = nn.Linear(2, 1)


    def forward(self, x1, x2):
        out1 = self.resnet1(x1)
        out2 = self.resnet2(x2)
        out = torch.concat((out1, out2), dim=-1)
        out = self.fc(out)
        return out