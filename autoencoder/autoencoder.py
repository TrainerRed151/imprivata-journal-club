import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self, hidden_layer_sizes=[100], learning_rate=1e-3, epochs=10, random_state=None):
        super().__init__()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.learning_rate = learning_rate
        self.epochs = epochs

        #if self.random_state:
        #    torch.random.manual_seed(random_state)

        self.encoder.append(nn.Linear(28*28, hidden_layer_sizes[0]))
        self.encoder.append(nn.ReLU())

        n = len(hidden_layer_sizes)
        for i in range(n - 1):
            self.encoder.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            self.encoder.append(nn.ReLU())

            self.decoder.append(nn.Linear(hidden_layer_sizes[n-i-1], hidden_layer_sizes[n-i-2]))
            self.decoder.append(nn.ReLU())

        self.decoder.append(nn.Linear(hidden_layer_sizes[0], 28*28))
        self.decoder.append(nn.Sigmoid())

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-8
        )

    def forward(self, X):
        encode = self.encoder(X)
        decode = self.decoder(encode)

        return decode

    def fit(self, loader):
        outputs = []
        losses = []
        for epoch in range(self.epochs):
            print(epoch)
            for image, _ in loader:
                image = image.reshape(-1, 28*28)

                reconstructed = self(image)

                loss = self.loss_function(reconstructed, image)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss)
                outputs.append((self.epochs, image, reconstructed))

        print(losses)

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.plot(losses[-100:])



if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=tensor_transform
    )

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    ae = AutoEncoder(hidden_layer_sizes=[784, 128, 64, 36, 18, 9])
    ae.fit(loader)
