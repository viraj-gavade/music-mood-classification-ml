import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class CNNEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*4*4, out_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

class MathMLP(nn.Module):
    def __init__(self, in_features=6, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class HybridModel(nn.Module):
    def __init__(self, num_classes=4, cnn_out=128, math_out=32):
        super().__init__()
        self.cnn = CNNEncoder(cnn_out)
        self.mlp = MathMLP(6, math_out)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out + math_out, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.apply(init_weights)

    def forward(self, mel, math_feats):
        a = self.cnn(mel)
        b = self.mlp(math_feats)
        x = torch.cat([a, b], dim=1)
        return self.classifier(x)
