import torch
from torch import Tensor, nn


def init_weights(l):
        """The function intializes the weights of a network with xavier initialization.
           Only for linear layers"""
        if isinstance(l, nn.Linear):
            torch.nn.init.xavier_normal_(l.weight) 
            l.bias.data.fill_(0.01)
            print(l.weight.shape, l.bias.shape)


class LightNet(nn.Module):
    """
    THe network consists of a convolutional autoencoder (pretrained) to extract image features.
    After pretraining, the MLP takes as input the features extracted by the encoder and returns SH coefficients representing
    the environment light.
    """
    def __init__(self, latent_dim: int = 24, kernel_size: int = 3, dense_layer_size: int = 64,
                 sh_degree: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.dense_layer_size = dense_layer_size
        self.sh_dim = (sh_degree + 1)**2
        self.latent_dim = latent_dim
    
        self.encoder = nn.Sequential(nn.conv2D(3, 64, self.kernel_size, padding='same'), #add batch normalization?
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.conv2D(64, 64, self.kernel_size, padding='same'),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.Conv2D(64, 128,  self.kernel_size, padding='same'),
                                     nn.ReLU(),
                                     nn.MaxPool2d
                                     )
        self.decoder = nn.Sequential(nn.convTranspose2d(128, 64,  self.kernel_size, padding='same'),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.convTranspose2d(64, 64,  self.kernel_size, padding='same'),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2),
                                     nn.convTranspose2d(64, 3,  self.kernel_size, padding='same'),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2))
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.dense_layer_size),
            nn.Dropout(p=0.5),
            nn.ReLU(), 
            nn.Linear(self.dense_layer_size, self.dense_layer_size),
            nn.ReLU(),
            nn.Linear(self.dense_layer_size, self.dense_layer_size),
            nn.ReLU()
        )
        self.sh_base = nn.Linear(self.dense_layer_size, 3)
        self.sh_rest = nn.Linear(self.dense_layer_size, (self.sh_dim - 1) * 3)
        #TODO: define the below layers
        # zero initialization for rest sh linear layers
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

        for linear_layer in [self.mlp, self.sh_base, self.sh_rest]:
             linear_layer.apply(init_weights)


    def forward(self, x, pretraining=False):
         encoded = self.encoder(x)
         if pretraining:
            decoded = self.decoder(encoded)
            return decoded
         else:
            encoded = nn.Flatten(encoded)
            assert encoded.shape[-1] == self.latent_dim, f" The expected latent dimension is {self.latent_dim}, got {encoded.shape[-1]} "
            sh_base = self.sh_base(self.mlp(encoded))
            sh_rest = self.sh_rest(self.mlp(encoded))
            sh_coeffs = torch.cat([sh_base, sh_rest], dim=-1).view(-1, self.sh_dim, 3)
            return sh_coeffs

        
        