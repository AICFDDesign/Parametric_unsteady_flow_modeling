"""
The archiecture for beta-VAE with para. branch NN
"""

import torch 
from torch import nn 


class parasVAE(nn.Module):
    def __init__(self, latent_dim, input_paradim):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_paradim = input_paradim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim)
        self.paracoder = self.buileParaCoder(latent_dim, input_paradim)

    def buildEncoder(self, latent_dim):
        encoder = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            #nn.ConstantPad3d((0, 0, 0, 1), 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.ConstantPad3d((0, 1, 0, 1), 0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.ConstantPad3d((0, 1, 0, 1), 0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.ConstantPad3d((0, 1, 0, 1), 0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Flatten(start_dim=1, end_dim=-1),

            nn.Linear(256*4*4, 256),
            nn.ELU(),

            nn.Linear(256, latent_dim * 2),
        )
        return encoder


    def buildDecoder(self, latent_dim):
        decoder = nn.Sequential(

            nn.Linear(latent_dim, 256),
            nn.ELU(),

            nn.Linear(256, 256 * 4 * 4),
            nn.ELU(),

            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConstantPad2d((0, -1, 0, -1), 0),
            nn.ELU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConstantPad2d((0, -1, 0, -1), 0),
            nn.ELU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConstantPad2d((0, -1, 0, -1), 0),
            nn.ELU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConstantPad2d((0, 0, 0, -1), 0),
            nn.ELU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ConstantPad2d((0, 0, 0, 0), 0),
            nn.ELU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),

        )
        return decoder
    def buileParaCoder(self, laten_dim, input_paradim):
        nodes = 128
        ParaCoder = nn.Sequential(
            nn.Linear(input_paradim, nodes),
            nn.ELU(),

            nn.Linear(nodes, nodes),
            nn.ELU(),

            nn.Linear(nodes, nodes),
            nn.ELU(),

            nn.Linear(nodes, nodes),
            nn.ELU(),

            nn.Linear(nodes, nodes),
            nn.ELU(),

            nn.Linear(nodes, nodes),
            nn.ELU(),

            nn.Linear(nodes, laten_dim*2),
            nn.ELU(),

        )
        return ParaCoder

    def sample(self, mean, logvariance):
        """
        Implementing reparameterlisation trick 
        """

        std = torch.exp(0.5 * logvariance)
        epsilon = torch.rand_like(std)

        return mean + epsilon*std

    def forward(self, data):
        w , para = data

        mean_logvariance = self.encoder(w)*self.paracoder(para)

        mean, logvariance = torch.chunk(mean_logvariance, 2, dim=1)

        z = self.sample(mean, logvariance)

        reconstruction = self.decoder(z)

        return reconstruction, mean, logvariance
