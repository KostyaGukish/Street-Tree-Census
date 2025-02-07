from torch import nn
from torch.nn import functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class TreeCensusModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(TreeCensusModel, self).__init__()

        self.layer1 = MLPBlock(input_dim, 256, dropout_prob)
        self.layer2 = MLPBlock(256, 512, dropout_prob)
        self.layer3 = MLPBlock(512, 512, dropout_prob)
        self.layer4 = MLPBlock(512, 1024, dropout_prob)
        self.layer5 = MLPBlock(1024, 512, dropout_prob)
        self.layer6 = MLPBlock(512, 256, dropout_prob)
        self.layer7 = MLPBlock(256, 256, dropout_prob)
        self.layer8 = MLPBlock(256, 64, dropout_prob)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.fc(x)
        return x
