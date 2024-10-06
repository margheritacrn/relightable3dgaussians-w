import torch
from torch import Tensor, nn
import numpy as np
from tqdm import tqdm
from data.dataloader import load_train_test
# TODO : add save_weights function, as well as load_weights and think it looks good how I implemented the class and the autoencoder thing.
# TODO: is np.int32 necessary?
# TODO: input resized to? 256? assume 3x256x256
# TODO: consider whether to add another dense layer to the encoder
# TODO: spostare pretrain_ae in lightNet.py, import from data serve solo a pretrain_ae

def init_weights(l):
        """The function intializes the weights of linear layers of a network
            using xavier initialization.
         """
        if isinstance(l, nn.Linear):
            torch.nn.init.xavier_normal_(l.weight) 
            l.bias.data.fill_(0.01)


class LightNet(nn.Module):
    """
    The network consists of a convolutional autoencoder (pretrained) to extract image features.
    After pretraining, the MLP takes as input the features extracted by the encoder and returns SH coefficients representing
    the environment light.
    """
    def __init__(self, latent_dim: int = 5, kernel_size: int = 3,
                 channels_f: int = 128,
                 dense_layer_size: int = 64, sh_degree: int = 4, input_shape: int =256):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels_f = channels_f
        self.dense_layer_size = dense_layer_size
        self.sh_dim = (sh_degree + 1)**2
        self.latent_dim = latent_dim
        self.ae_loss = torch.nn.MSELoss
        self.input_shape = input_shape
        enc_end_shape = self.channels_f*(np.int32(input_shape/4)**2)
        self.ae_optimizer = torch.optim.Adam
    
        self.encoder_conv = nn.Sequential(nn.Conv2d(3, np.int32(self.channels_f/2), self.kernel_size, padding='same'), # 3x256x256 -> 64x256x256
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.Conv2d(np.int32(self.channels_f/2), np.int32(self.channels_f/2), self.kernel_size, padding='same'),
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.AvgPool2d(kernel_size=2), #64x128x128
                                     nn.Conv2d(np.int32(self.channels_f/2), self.channels_f,  self.kernel_size, padding='same'),
                                     nn.BatchNorm2d(self.channels_f),
                                     nn.ReLU(),
                                     nn.Conv2d(self.channels_f, self.channels_f,  self.kernel_size, padding='same'), #128x128x128
                                     nn.BatchNorm2d(self.channels_f),
                                     nn.ReLU(),
                                     nn.AvgPool2d(kernel_size=2) # 128x64x64
                                    )
        self.encoder_dense = nn.Linear(enc_end_shape, 2**self.latent_dim)

        self.decoder_conv = nn.Sequential(
                                     nn.ConvTranspose2d(self.channels_f, self.channels_f,  self.kernel_size, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(self.channels_f),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(self.channels_f, np.int32(self.channels_f/2), self.kernel_size, padding=1),
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(np.int32(self.channels_f/2), np.int32(self.channels_f/2),  self.kernel_size, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(np.int32(self.channels_f/2), 3, self.kernel_size, padding=1),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU()
                                     )
        self.decoder_dense = nn.Linear(2**self.latent_dim, enc_end_shape)

        self.mlp = nn.Sequential(
            nn.Linear(2**self.latent_dim, self.dense_layer_size),
            nn.Dropout(p=0.2),
            nn.ReLU(), 
            nn.Linear(self.dense_layer_size, self.dense_layer_size),
            nn.ReLU(),
            nn.Linear(self.dense_layer_size, self.dense_layer_size),
            nn.ReLU()
        )

        self.sh_base = nn.Linear(self.dense_layer_size, 3)
        self.sh_rest = nn.Linear(self.dense_layer_size, (self.sh_dim - 1) * 3)
        # zero initialization for rest sh linear layers
        self.sh_rest.weight.data.zero_()
        self.sh_rest.bias.data.zero_()

        for linear_layer in [self.encoder_dense, self.decoder_dense, self.mlp, self.sh_base]:
             linear_layer.apply(init_weights)


    def forward(self, x, pretraining=False):
         x_conv = self.encoder_conv(x)
         feature_vec = self.encoder_dense(torch.flatten(x_conv, start_dim=1))
         if pretraining:
            decoded = self.decoder_dense(feature_vec)
            decoded = torch.unflatten(decoded, dim=1, sizes=(self.channels_f, np.int32(self.input_shape/4), np.int32(self.input_shape/4)))
            decoded = self.decoder_conv(decoded)
            return decoded
         else:
            sh_base = self.sh_base(self.mlp(feature_vec))
            sh_rest = self.sh_rest(self.mlp(feature_vec))
            sh_coeffs = torch.cat([sh_base, sh_rest], dim=-1).view(-1, self.sh_dim, 3)
            return sh_coeffs


    def get_optimizer(self):
        return self.ae_optimizer(self.parameters(), lr=0.001, weight_decay=1e-5)


    def save_weights(self, path: str, epoch: int):
        torch.save(self.state_dict(), path + "/lightNetAE_weights_epoch_"+str(epoch)+".pth")

   
    def pretrain_ae(self, data_path, num_epochs: int = 50, resize_dim: int = 256, batch_size: int = 32,
                    verbose=False, tensorboard_writer=None,
                    progress_bar=None, output_path=None, return_losses=False,
                    return_data_transf=False):
        self.cuda()
        loss_ = self.ae_loss()
        losses_tr_all = []
        losses_test_all = []
        optim = self.get_optimizer()

        for epoch in range(num_epochs):
            self.train()
            losses_tr = []
            train_iter, test_iter, data_transforms = load_train_test(datapath=data_path, resize_dim=resize_dim, batch_size=batch_size)

            for batch in train_iter:
                optim.zero_grad() 
                input = batch.cuda()
                reconstructed_input = self(input, pretraining=True)
                mse = loss_(reconstructed_input, input)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step() 
                losses_tr.append(mse.detach().cpu().item())
            losses_tr_all.append(np.mean(losses_tr))
            if (tensorboard_writer):
                tensorboard_writer.add_scalar('train loss- ae',
                                losses_tr_all[epoch],
                                epoch)

            self.eval()
            losses_test = []
            with torch.no_grad():
                for batch in test_iter:
                    input = batch.cuda()
                    reconstructed_input = self(input, pretraining=True)
                    mse = loss_(reconstructed_input, input)
                    losses_test.append(mse.detach().cpu().item())
            torch.cuda.empty_cache()
            losses_test_all.append(np.mean(losses_test))
            if tensorboard_writer:
                tensorboard_writer.add_scalar('test loss- ae',
                            losses_test_all[epoch],
                            epoch)
            if verbose:
                print(f"Epoch: {epoch}, train loss: {losses_tr_all[epoch]}, test_loss: {losses_test_all[epoch]}")
            elif progress_bar:
                if epoch % 1 == 0:
                    progress_bar.set_postfix({"Loss test": f"{losses_test_all[epoch]:.{4}f}"})
                    progress_bar.update(1)
                if epoch == num_epochs -1:
                    progress_bar.close()

        if output_path:
            self.save_weights(output_path, epoch)
        
        if return_losses:
            out = [losses_test_all, losses_tr_all]
            if return_data_transf:
                out.append(data_transforms)
                return out
            else:
                return out
        elif return_data_transf:
            return data_transforms
        else:
            pass
