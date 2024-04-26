import torch
import torch.nn as nn
#from torchvision import datasets

class Autoencoder(nn.Module):
    def __init__(self, hidden_layer_sizes=[100], learning_rate=1e-3, max_iter=1000, batch_size=1000, random_state=None):
        super().__init__()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size

        if self.random_state:
            torch.random.manual_seed(random_state)

    def forward(self, X):
        encode = self.encoder(X)
        decode = self.decoder(encode)

        return decode

    def fit(self, X):
        self.encoder.append(nn.Linear(X.size[1], hidden_layer_size[0]))
        self.encoder.append(nn.ReLU())

        n = len(hidden_layer_sizes)
        for i in range(n - 1):
            self.encoder.append(nn.Linear(hidden_layer_size[i], hidden_layer_size[i+1]))
            self.encoder.append(nn.ReLU())

            self.decoder.append(nn.Linear(hidden_layer_size[n-i-1], hidden_layer_size[n-i-2]))
            self.decoder.append(nn.ReLU())

        self.decoder.append(hidden_layer_size[0], x.size[1])
        self.decoder.append(nn.Sigmoid())
