# print("Importing model...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

class TBCNN(nn.Module):
    def __init__(self):
        super(TBCNN, self).__init__()
        self.conv1 = nn.Conv2d(1,16, kernel_size = 3, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(16,32, kernel_size = 3, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(32,128, kernel_size = 3, stride = 1, padding = 0)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,32)
        # self.fc3 = nn.Linear(64,32)

        self.fc4 = nn.Linear(32,1)
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        
        x = self.gap(x)
        x = self.flatten(x)
        # x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        
        x = self.fc4(x)
        return x

if __name__ == "__main__":
    model = TBCNN()
    print ("Model architecture:")
    summary(model, input_size=(1, 224, 224))
    print("Model defined successfully")
