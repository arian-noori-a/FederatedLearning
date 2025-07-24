import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Extracts image embeddings."""
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc_embed = nn.Linear(64 * 8 * 8, embedding_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        embed = self.fc_embed(x)
        return embed


class Classifier(nn.Module):
    """Classifier head for embeddings."""
    def __init__(self, embedding_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc2 = nn.Linear(embedding_dim // 2, num_classes)

    def forward(self, embed):
        x = F.relu(self.fc1(embed))
        return self.fc2(x)


class Decoder(nn.Module):
    """Decoder/generator mapping embeddings back to images."""
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, 64 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

    def forward(self, embed):
        x = self.fc(embed).view(-1, 64, 8, 8)
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))


class EmbeddingClassifierDecoder(nn.Module):
    """
    Full model combining encoder, classifier, and decoder.
    - Encoder: produces embeddings for clustering/GAN input
    - Classifier: for FedAvg supervised training
    - Decoder: for GAN augmentation
    """
    def __init__(self, embedding_dim=128, num_classes=10):
        super().__init__()
        self.encoder = Encoder(embedding_dim)
        self.classifier = Classifier(embedding_dim, num_classes)
        self.decoder = Decoder(embedding_dim)

    def forward(self, x, mode='classifier'):
        embed = self.encoder(x)
        if mode == 'classifier':
            return self.classifier(embed)
        elif mode == 'decoder':
            return self.decoder(embed)
        elif mode == 'embed':
            return embed
        else:
            raise ValueError(f"Invalid mode: {mode}")
