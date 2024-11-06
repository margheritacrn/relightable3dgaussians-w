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
import numpy as np
from utils.system_utils import mkdir_p
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image


@hydra.main(version_base=None, config_path="../configs", config_name="relightable3DG-W")
class Relightable3DGW:
    def __init__(self,  config: DictConfig, checkpoint: str = None):
        self.config = config # OmegaConf.structured(OmegaConf.to_yaml(config))
        self.checkpoint = checkpoint
        self.optimizer: torch.optim = None
        self.iteration: int = 0

        self.gaussians: GaussianModel = GaussianModel(self.config.sh_degree)
        self.scene:  Scene = Scene(self.config.dataset, self.gaussians)
        self.train_cameras = self.scene.getTrainCameras().copy()
        if self.config.num_sky_points > 0:
            self.gaussians.extend_with_sky_gaussians(num_points = self.config.num_sky_points, cameras = self.train_cameras)

        self.envlight_sh_mlp: SHMlp = SHMlp(sh_degree = self.config.envlight_sh_degree, embedding_dim=self.config.embeddings_dim)
        self.envlight_sh_mlp.cuda()
        if self.config.optimizer.lambda_envlight_sh_prior > 0 or self.config.init_sh_mlp:
            if os.path.exists(self.config.envlight_sh_prior_path):
                self.envlight_sh_priors = load_npy_tensors(Path(self.config.envlight_sh_prior_path))
            else:
                raise ValueError('Path for environment light sh priors not found')
        
        self.envlight = EnvironmentLight(base = torch.empty(((self.config.envlight_sh_degree +1)**2), 3),
                                         sh_degree=self.config.envlight_sh_degree)
        self.embeddings = torch.nn.Embedding(len(self.train_cameras),
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
        embeddings_inits = torch.zeros((len(viewpoint_stack), 32), dtype=torch.float32,  device='cuda')
        with torch.no_grad():
            batch_imgs_1 = torch.stack([data_transforms(viewpoint_cam.original_image) for viewpoint_cam in viewpoint_stack[:int(len(viewpoint_stack)/2)]]).to(dtype=torch.float32,  device='cuda')
            embeddings_inits[:int(len(viewpoint_stack)/2)] = embedding_network(batch_imgs_1)
            del batch_imgs_1
            batch_imgs_2 = torch.stack([data_transforms(viewpoint_cam.original_image) for viewpoint_cam in viewpoint_stack[int(len(viewpoint_stack)/2):]]).to(dtype=torch.float32,  device='cuda')
            embeddings_inits[int(len(viewpoint_stack)/2):] = embedding_network(batch_imgs_2)
            del batch_imgs_2
        del embedding_network
        torch.cuda.empty_cache()
        # Initialize per-image embeddings
        self.embeddings.weight = torch.nn.Parameter(F.normalize(embeddings_inits, p=2, dim=-1))

    def initialize_sh_mlp(self):
        print("Initializing SH MLP")
        viewpoint_stack = self.scene.getTrainCameras().copy()
        imgs = torch.stack([self.embeddings(torch.tensor([viewpoint_cam.uid], device = 'cuda')).detach() for viewpoint_cam in viewpoint_stack]).to(dtype=torch.float32,  device='cuda')
        lighting_conditions = [viewpoint_cam.image_name[:-9] for viewpoint_cam in viewpoint_stack]
        sh_priors_keys = [next((key for key in self.envlight_sh_priors if lc in key), None) for lc in lighting_conditions]
        target_sh = torch.stack([torch.tensor(self.envlight_sh_priors[key], dtype=torch.float32) for key in sh_priors_keys ])
        train_data = TensorDataset(imgs, target_sh)
        batch_size = 32
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.envlight_sh_mlp.initialize(dataloader, epochs = 100)


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


    def get_envlights_sh_all(self):
        envlights_sh = {}
        viewpoint_stack = self.scene.getTrainCameras().copy()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for viewpoint_cam in viewpoint_stack:
                viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device = 'cuda')
                image_embed = self.embeddings(viewpoint_cam_id)
                envlights_sh[viewpoint_cam.image_name] = self.envlight_sh_mlp(image_embed).detach().cpu().numpy()
        return envlights_sh
    

    def render_envlights_sh_all(self, save_path: str):
        envlights_sh = self.get_envlights_sh_all()
        for im_name in envlights_sh.keys():
            self.envlight.set_base(envlights_sh[im_name])
            rendered_sh = self.envlight.render_sh()
            rendered_img = Image.fromarray(rendered_sh)
            save_path = os.path.joint(save_path, im_name + ".jpg")
            rendered_img.save(save_path)


    def save_config(self):
        config_path = os.path.join(self.config.dataset.model_path, "relightable3DG-W_run.yaml")
        with open(config_path, "w") as yaml_file:
            OmegaConf.save(self.config, yaml_file)


    def save(self, iteration: int):
        model_path = self.config.dataset.model_path
        embeds_path = os.path.join(model_path, "checkpoint_embeddings/iteration_{}".format(iteration))
        os.makedirs(embeds_path, exist_ok=True)
        # mkdir_p(os.path.dirname(os.path.join(embeds_path, "embeddings_weights.pth")))
        envlights_sh_path = os.path.join(model_path, "envlights_sh/iteration_{}".format(iteration))
        os.makedirs(envlights_sh_path, exist_ok=True)
        #mkdir_p(os.path.dirname(envlights_sh_path))
        sh_mlp_path = os.path.join(model_path, "checkpoint_SHMlp/iteration_{}".format(iteration))
        os.makedirs(sh_mlp_path, exist_ok=True)
        #mkdir_p(os.path.dirname(os.path.join(sh_mlp_path, "SHMlp_weights.pth")))
        print("Saving embeddings weights\n")
        torch.save(self.embeddings.weight, embeds_path + "/embeddings_weights.pth")
        print("Saving SH MLP weights\n")
        torch.save(self.envlight_sh_mlp.state_dict(), sh_mlp_path +  "/SHMlp_weights.pth")
        torch.save
        print("Saving envlights SH coefficients\n")
        envlights_sh = self.get_envlights_sh_all()
        for envlight_sh in envlights_sh.keys():
            np.save(envlights_sh_path + "/envlight_sh_" + envlight_sh + ".npy", envlights_sh[envlight_sh])
        print("Saving Gaussians\n")
        self.scene.save(iteration)


    def load_model(self, iteration: int):
        pass
            





