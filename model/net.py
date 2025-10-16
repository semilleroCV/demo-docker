import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(84, 10)
        
        # Apply proper weight initialization (kaiming for ReLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using best practices from llm.txt"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

