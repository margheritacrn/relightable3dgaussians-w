import torch
from scene.NVDIFFREC import EnvironmentLight
from scene.net_models import SHMlp, EmbeddingNet
from omegaconf import OmegaConf, DictConfig
import hydra
from scene import GaussianModel, Scene
from utils.general_utils import load_npy_tensors
import os
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F


@hydra.main(version_base=None, config_path="../configs", config_name="relightable3DG-W")
class Relightable3DGW:
    def __init__(self,  config: DictConfig, checkpoint: str = None):
        self.config = config # OmegaConf.structured(OmegaConf.to_yaml(config))
        self.checkpoint = checkpoint
        self.optimizer: torch.optim = None
        self.iteration: int = 0

        self.gaussians: GaussianModel = GaussianModel(self.config.sh_degree)

        self.scene:  Scene = Scene(self.config.dataset, self.gaussians)

        self.envlight_sh_mlp: SHMlp = SHMlp(sh_degree = self.config.envlight_sh_degree, embedding_dim=self.config.embeddings_dim)
        self.envlight_sh_mlp.cuda()
        if self.config.optimizer.lambda_envlight_sh_prior > 0:
            if os.path.exists(self.config.envlight_sh_prior_path):
                self.envlight_sh_priors = load_npy_tensors(Path(self.config.envlight_sh_prior_path))
            else:
                raise ValueError('Path for environment light sh priors not found')
        
        self.envlight = EnvironmentLight(base = torch.empty(((self.config.envlight_sh_degree +1)**2), 3),
                                         sh_degree=self.config.envlight_sh_degree)
        self.embeddings = torch.nn.Embedding(len(self.scene.train_cameras[1.0]),
                                             self.config.embeddings_dim)
        self.embeddings.cuda()
        if self.config.init_embeddings:
            self.initialize_embeddings()

        self.save_config()


    def initialize_embeddings(self):
        embednet_pretrain_epochs = self.config.optimizer.embednet_pretrain_epochs
        viewpoint_stack = self.scene.getTrainCameras().copy()
        embedding_network = EmbeddingNet(latent_dim=self.config.embeddings_dim)
        embedding_network.cuda()
        checkpoint = self.config.dataset.model_path + f'/EmbeddingNet_model_epoch_{embednet_pretrain_epochs-1}.pth'

        if os.path.exists(checkpoint):
            print(f"EmbeddingNet- Loading pretrained weights")
            state_dict = torch.load(checkpoint, weights_only=True)
            embedding_network.load_state_dict(state_dict)
            data_transforms = embedding_network.optimize(data_path=self.config.dataset.source_path, get_datatransforms_only=True)
        else:
            print("EmbeddingNet- Start training")
            progress_bar_light_ae = tqdm(range(1, embednet_pretrain_epochs + 1), desc = "Training images embeddings pretraining progress")
            output_embednet = embedding_network.optimize(data_path=self.config.dataset.source_path,
                                                                    num_epochs = embednet_pretrain_epochs,
                                                                    progress_bar=progress_bar_light_ae,
                                                                    output_path=self.config.dataset.model_path, return_outputs=True
                                                                )
            data_transforms = output_embednet["data_transforms"]

        embedding_network.eval()
        with torch.no_grad():
            imgs = [data_transforms(viewpoint_cam.original_image) for viewpoint_cam in viewpoint_stack]
            batch_imgs = torch.stack(imgs).to(dtype=torch.float32,  device='cuda')
            embeddings_inits = embedding_network(batch_imgs)
        del embedding_network
        torch.cuda.empty_cache()
        # Initialize per-image embeddings
        self.embeddings.weight = torch.nn.Parameter(F.normalize(embeddings_inits, p=2, dim=-1))


    def training_set_up(self):
        training_args = self.config.optimizer
        gaussians_opt_params = self.gaussians.training_setup_relit3DGW(training_args)

        model_opt_params =  [
            {'params': self.envlight_sh_mlp.parameters(), 'lr': training_args.envlight_sh_lr,
             'weight_decay': training_args.envlight_sh_wd, "name": 'envlight_sh'},
            {'params': self.embeddings.parameters(), 'lr': training_args.embeddings_lr, "name": 'embeddings'}
        ]
        model_opt_params.extend(gaussians_opt_params)

        self.optimizer = torch.optim.Adam(model_opt_params, lr=0.01, eps=1e-15)
        self.gaussians.set_optimizer(self.optimizer)

   
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self.gaussians.update_learning_rate(iteration)
        for param_group in self.optimizer.param_groups:
            if iteration == 30000 and (param_group["name"] == "envlight_sh" or param_group["name"] == "embeddings"):
                param_group['lr'] = 0.0001
    

    def save_config(self):
        config_path = os.path.join(self.config.dataset.model_path, "relightable3DG-W_all.yaml")
        with open(config_path, "w") as yaml_file:
            OmegaConf.save(self.config, yaml_file)




