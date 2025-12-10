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
        # Automatically determine in_features for self.fc
        dummy = torch.zeros(1, 1, 32, 32)  # Adjust shape as needed for your input
        with torch.no_grad():
            dummy_out = self.conv(dummy)
            flat_dim = dummy_out.flatten(1).shape[1]
        self.fc = nn.Linear(flat_dim, out_dim)
        self.apply(init_weights)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device=device, dtype=torch.float32)
        print(f"Input shape: {x.shape}")
        x = self.conv(x)
        print(f"After conv shape: {x.shape}")
        x = self.flatten(x)
        print(f"After flatten shape: {x.shape}")
        x = self.fc(x)
        print(f"After fc shape: {x.shape}")
        x = F.relu(x)
        print(f"After relu shape: {x.shape}")
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
        device = next(self.parameters()).device
        x = x.to(device=device, dtype=torch.float32)
        print(f"MathMLP input shape: {x.shape}")
        out = self.net(x)
        print(f"MathMLP output shape: {out.shape}")
        return out

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
        device = next(self.parameters()).device
        mel = mel.to(device=device, dtype=torch.float32)
        math_feats = math_feats.to(device=device, dtype=torch.float32)
        print(f"HybridModel mel shape: {mel.shape}")
        print(f"HybridModel math_feats shape: {math_feats.shape}")
        a = self.cnn(mel)
        b = self.mlp(math_feats)
        x = torch.cat([a, b], dim=1)
        print(f"HybridModel concat shape: {x.shape}")
        out = self.classifier(x)
        print(f"HybridModel output shape: {out.shape}")
        return out
