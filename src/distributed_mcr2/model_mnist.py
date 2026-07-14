import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_MNIST(nn.Module):

    def __init__(self, dim_z, model=None):
        super().__init__()

        self.dim_z = dim_z
        ndf=64
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, dim_z, 4, 1, 0, bias=False),
            nn.Flatten()  # new
            # nn.LeakyReLU(0.2, inplace=True), #New
            # nn.Linear(ndf, ndf, bias=False)
            # nn.Sigmoid()
        )

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        return F.normalize(self.main(x))


class Classifier(nn.Module):

    def __init__(self, num_classes, dim_z):
        super().__init__()

        # print("DEBUG: modality is: ", modality)
        
        self.dim_z = dim_z
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_z, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, num_classes)
        )
        '''
        self.classifier = nn.Linear(self.dim_z, num_classes)'''


    def forward(self, feature):
        # print(x.shape)
        output = self.classifier(feature)

        return output


class Encoder_VAE(nn.Module):

    def __init__(self, dim_z):
        super().__init__()

        self.dim_z = dim_z
        ndf=64
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, dim_z, 4, 1, 0, bias=False),
            nn.Flatten()  # new
            # nn.LeakyReLU(0.2, inplace=True), #New
            # nn.Linear(ndf, ndf, bias=False)
            # nn.Sigmoid()
        )

        self.fc_mu = nn.Linear(128, dim_z)
        self.fc_logvar = nn.Linear(128, dim_z)

    def forward(self, x):
        h = self.main(x)
        # print(h.shape)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class JointDecoder_VAE(nn.Module):
    def __init__(self, latent_dim, num_agents, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * num_agents, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, zs):
        z = torch.cat(zs, dim=1)
        return self.net(z)