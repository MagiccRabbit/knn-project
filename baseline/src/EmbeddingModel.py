import torch
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, n_mels=80, embedding_dim=192, num_speakers=100):
        super().__init__()

        self.conv_shortcut = nn.Conv1d(n_mels, 512, kernel_size=1)

        self.conv1 = nn.Conv1d(n_mels, 512, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv4 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        self.relu = nn.ReLU()

        # statistics pooling (mean + std)
        self.fc = nn.Linear(512 * 2, embedding_dim)

        #dropout
        self.dropout_conv = nn.Dropout1d(p=0.3)
        self.dropout_pooling = nn.Dropout(p=0.3)

        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def forward(self, x):
        # x: (batch, n_mels, time)
    
        identity = self.conv_shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_conv(x) + identity

        identity = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_conv(x) + identity

        identity = x
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x) + identity

        identity = x
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout_conv(x) + identity

        # pooling
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        x = torch.cat([mean, std], dim=1)

        x = self.dropout_pooling(x)
        emb = self.fc(x)

        logits = self.classifier(emb)

        return emb, logits