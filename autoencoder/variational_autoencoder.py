import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, hidden_layer_sizes=[100], latent_dim=20, learning_rate=1e-3, epochs=10, random_state=None):
        super().__init__()

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.latent_dim = latent_dim

        # Encoder
        self.encoder.append(nn.Linear(28*28, hidden_layer_sizes[0]))
        self.encoder.append(nn.ReLU())
        for i in range(len(hidden_layer_sizes) - 1):
            self.encoder.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            self.encoder.append(nn.ReLU())

        # Latent space
        self.fc_mu = nn.Linear(hidden_layer_sizes[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layer_sizes[-1], latent_dim)

        # Decoder
        self.decoder.append(nn.Linear(latent_dim, hidden_layer_sizes[-1]))
        self.decoder.append(nn.ReLU())
        for i in range(len(hidden_layer_sizes) - 1, 0, -1):
            self.decoder.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i-1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(hidden_layer_sizes[0], 28*28))
        self.decoder.append(nn.Sigmoid())

        self.loss_function = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-8
        )

    def encode(self, X):
        h = self.encoder(X)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, X):
        mu, logvar = self.encode(X)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def fit(self, loader):
        outputs = []
        losses = []
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            for image, _ in loader:
                image = image.view(-1, 28*28)

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self(image)
                loss = self.loss_function(recon_batch, image, mu, logvar)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                outputs.append((epoch, image, recon_batch))

            print(f'Loss: {loss.item()}')

        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.plot(losses)
        plt.show()


if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=tensor_transform
    )

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    vae = VAE(hidden_layer_sizes=[784, 128, 64], latent_dim=20)
    vae.fit(loader)
