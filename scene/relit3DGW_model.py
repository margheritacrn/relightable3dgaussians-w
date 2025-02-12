import torch
from scene.NVDIFFREC import EnvironmentLight
from scene.net_models import SHMlp, EmbeddingNet
from omegaconf import OmegaConf, DictConfig
import hydra
from scene import GaussianModel, Scene
from utils.general_utils import load_npy_tensors
from utils.system_utils import searchForMaxIteration
import os
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from utils.system_utils import mkdir_p
from utils.general_utils import get_half_images
from torch.utils.data import TensorDataset, DataLoader
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, sky_depth_loss
from PIL import Image
from random import randint


@hydra.main(version_base=None, config_path="../configs", config_name="relightable3DG-W")
class Relightable3DGW:
    def __init__(self,  config: DictConfig):
        self.config = config
        self.load_iteration = self.config.load_iteration
        self.optimizer: torch.optim = None
        if self.load_iteration != 'None':
            outputs_paths = ["point_cloud", "checkpoint_embeddings", "checkpoint_SHMlp", "envlights_sh"]
            if self.load_iteration == -1:
                outputs_paths = ["point_cloud", "checkpoint_embeddings", "checkpoint_SHMlp", "envlights_sh"]
                load_iters = [searchForMaxIteration(os.path.join(self.config.dataset.model_path, op)) for op in outputs_paths]
                assert len(set(load_iters)) == 1, f"Load iteration- incongruous number of saved iterations"
                self.load_iteration = load_iters[0]
            else:
                for op in outputs_paths:
                    assert os.path.exists(os.path.join(self.config.dataset.model_path, op+f"/iteration_{self.load_iteration}")), f"Load iteration {self.load_iteration}- output path missing"
                    self.load_iteration = self.load_iteration
            self.gaussians: GaussianModel = GaussianModel()
            self.scene:  Scene = Scene(self.config.dataset, self.gaussians, load_iteration=self.load_iteration)
            self.train_cameras = self.scene.getTrainCameras().copy()
            self.test_cameras = self.scene.getTestCameras().copy()
            self.load_model()
        else:
            self.gaussians: GaussianModel = GaussianModel()
            self.scene:  Scene = Scene(self.config.dataset, self.gaussians)
            self.train_cameras = self.scene.getTrainCameras().copy()
            self.test_cameras = self.scene.getTestCameras().copy()
            if self.config.num_sky_points > 0:
                self.gaussians.augment_with_sky_gaussians(num_points = self.config.num_sky_points, cameras = self.train_cameras)
            # self.scene.cameras_extent = self.gaussians.get_scene_extent(self.train_cameras)
            self.envlight_sh_mlp: SHMlp = SHMlp(sh_degree = self.config.envlight_sh_degree, embedding_dim=self.config.embeddings_dim)
            self.envlight_sh_mlp.cuda()
            if self.config.optimizer.lambda_envlight_sh_prior > 0 or self.config.init_sh_mlp:
                self.envlight_sh_prior_path = os.path.join(self.config.dataset.source_path, "train/envmaps_init")
                if os.path.exists(self.envlight_sh_prior_path):
                    self.envlight_sh_priors = load_npy_tensors(Path(self.envlight_sh_prior_path))
                else:
                    raise ValueError('Path for environment light sh priors not found')
            
            self.embeddings = torch.nn.Embedding(len(self.train_cameras),
                                                self.config.embeddings_dim)
            self.embeddings.cuda()
            if self.config.init_embeddings:
                self.initialize_embeddings()
        
        self.envlight = EnvironmentLight(base = torch.empty(((self.config.envlight_sh_degree +1)**2), 3),
                                            sh_degree=self.config.envlight_sh_degree)
        self.embeddings_test = None

        if not self.config.dataset.eval:
            self.save_config()


    def initialize_embeddings(self, test = False):
        embednet_pretrain_epochs = self.config.optimizer.embednet_pretrain_epochs
        viewpoint_stack = self.train_cameras
        if test:
            viewpoint_stack = self.test_cameras
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
        if test:
            self.embeddings_test.weight = torch.nn.Parameter(F.normalize(embeddings_inits, p=2, dim=-1))
        else:
            self.embeddings.weight = torch.nn.Parameter(F.normalize(embeddings_inits, p=2, dim=-1))


    def initialize_sh_mlp(self):
        print("Initializing SH MLP")
        viewpoint_stack = self.scene.getTrainCameras().copy()
        imgs = torch.stack([self.embeddings(torch.tensor([viewpoint_cam.uid]).to(dtype=torch.long, device='cuda')).detach() for viewpoint_cam in viewpoint_stack]).to(dtype=torch.float32,  device='cuda')
        lighting_conditions = [viewpoint_cam.image_name[:-9] if viewpoint_cam.image_name[0] != "C" else viewpoint_cam.image_name[:3] for viewpoint_cam in viewpoint_stack]
        sh_priors_keys = [next((key for key in self.envlight_sh_priors if lc in key), None) for lc in lighting_conditions]
        target_sh = torch.stack([torch.tensor(self.envlight_sh_priors[key], dtype=torch.float32) for key in sh_priors_keys ])
        train_data = TensorDataset(imgs, target_sh[:, :(self.config.envlight_sh_degree + 1)**2, :])
        batch_size = 32
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.envlight_sh_mlp.initialize(dataloader, epochs = 100)


    def training_set_up(self):
        training_args = self.config.optimizer
        gaussians_opt_params = self.gaussians.training_setup(training_args)

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


    def get_envlights_sh_all(self, eval=False):
        envlights_sh = {}
        if eval:
            viewpoint_stack = self.scene.getTestCameras().copy()
        else:
            viewpoint_stack = self.scene.getTrainCameras().copy()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for viewpoint_cam in viewpoint_stack:
                viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device = 'cuda')
                if eval:
                    image_embed = self.embeddings_test(viewpoint_cam_id)
                else:
                    image_embed = self.embeddings(viewpoint_cam_id)
                envlights_sh[viewpoint_cam.image_name] = self.envlight_sh_mlp(image_embed).detach().cpu().numpy()
        return envlights_sh
    

    def render_envlights_sh_all(self, save_path: str, eval=False, save_sh_coeffs=False):
        envlights_sh = self.get_envlights_sh_all(eval)
        for im_name in envlights_sh.keys():
            self.envlight.set_base(envlights_sh[im_name])
            if save_sh_coeffs:
                np.save(os.path.join(save_path, im_name + ".npy"), self.envlight.base)
            rendered_sh = self.envlight.render_sh()
            rendered_img = Image.fromarray(rendered_sh)
            save_path_im = os.path.join(save_path, im_name + ".jpg")
            rendered_img.save(save_path_im)


    def save_config(self):
        config_path = os.path.join(self.config.dataset.model_path, "relightable3DG-W_run.yaml")
        with open(config_path, "w") as yaml_file:
            OmegaConf.save(self.config, yaml_file)


    def save(self, iteration: int):
        model_path = self.config.dataset.model_path
        embeds_path = os.path.join(model_path, "checkpoint_embeddings/iteration_{}".format(iteration))
        os.makedirs(embeds_path, exist_ok=True)
        envlights_sh_path = os.path.join(model_path, "envlights_sh/iteration_{}".format(iteration))
        os.makedirs(envlights_sh_path, exist_ok=True)
        sh_mlp_path = os.path.join(model_path, "checkpoint_SHMlp/iteration_{}".format(iteration))
        os.makedirs(sh_mlp_path, exist_ok=True)
        print("Saving embeddings weights\n")
        torch.save(self.embeddings.weight, embeds_path + "/embeddings_weights.pth")
        print("Saving SH MLP weights\n")
        torch.save(self.envlight_sh_mlp.state_dict(), sh_mlp_path +  "/SHMlp_weights.pth")
        print("Saving envlights SH coefficients\n")
        envlights_sh = self.get_envlights_sh_all()
        for envlight_sh in envlights_sh.keys():
            np.save(envlights_sh_path + "/envlight_sh_" + envlight_sh + ".npy", envlights_sh[envlight_sh])
        print("Saving Gaussians\n")
        self.scene.save(iteration)


    def load_model(self):
        """Load model at iteration self.load_iteration"""
        checkpoint_ply = os.path.join(self.config.dataset.model_path, f"point_cloud/iteration_{self.load_iteration}/point_cloud.ply")
        checkpoint_embeddings = os.path.join(self.config.dataset.model_path, f"checkpoint_embeddings/iteration_{self.load_iteration}/embeddings_weights.pth")
        checkpoint_sh_mlp = os.path.join(self.config.dataset.model_path, f"checkpoint_SHMlp/iteration_{self.load_iteration}/SHMlp_weights.pth")
        assert os.path.exists(checkpoint_ply) , f"Loading model at iter {self.load_iteration}- point cloud checkpoint path not found"
        assert  os.path.exists(checkpoint_embeddings), f"Loading model at iter {self.load_iteration}- embeddings checkpoint path not found"
        assert  os.path.exists(checkpoint_sh_mlp), f"Loading model at iter {self.load_iteration}- SH MLP checkpoint path not found" 

        self.gaussians.load_ply(checkpoint_ply)
                                         
        embeds = torch.load(checkpoint_embeddings, weights_only=True)
        self.embeddings = torch.nn.Embedding(len(self.train_cameras), self.config.embeddings_dim)
        self.embeddings.weight = embeds
        self.embeddings.cuda()

        state_dict_sh_mlp = torch.load(checkpoint_sh_mlp, weights_only=True)
        self.envlight_sh_mlp = SHMlp(sh_degree = self.config.envlight_sh_degree, embedding_dim=self.config.embeddings_dim)
        self.envlight_sh_mlp.load_state_dict(state_dict_sh_mlp)
        if self.config.dataset.eval:
            self.envlight_sh_mlp.eval()
        self.envlight_sh_mlp.cuda()


    def optimize_embeddings_test(self, mse=False):
        "Optimization of the images embeddings for the test set. Optimization is performed on left half of the test images."
        print(f"Optimizing test images embeddings on the left half of each image")
        # Initialize test embeddings:
        self.embeddings_test = torch.nn.Embedding(len(self.test_cameras),
                                                self.config.embeddings_dim)
        self.embeddings_test.cuda()
        optimizer = torch.optim.Adam(self.embeddings_test.parameters(), lr=self.config.optimizer.embeddings_lr)
        self.initialize_embeddings(eval=True)
        viewpoint_stack = self.scene.getTestCameras().copy()
        background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        mse_loss = torch.nn.MSELoss()
        with torch.enable_grad():
            for _ in tqdm(range(self.config.optimizer.optim_embeddings_test_iters), desc = "Test images embeds optimization"):
                if not viewpoint_stack:
                    viewpoint_stack = self.scene.getTestCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                viewpoint_cam_id = torch.tensor([viewpoint_cam.uid], device = 'cuda')
                gt_image = viewpoint_cam.original_image.cuda()
                sky_mask = viewpoint_cam.sky_mask.cuda().expand_as(gt_image)
                occluders_mask = viewpoint_cam.occluders_mask.cuda().expand_as(gt_image)
                gt_image_half = get_half_images(img = gt_image, left = True)
                occluders_mask_half = get_half_images(img = occluders_mask, left = True)
                sky_mask_half = get_half_images(img = sky_mask, left = True)
                # Update width:
                # viewpoint_cam.image_width = viewpoint_cam.image_width // 2
                embedding_gt_image_half = self.embeddings_test(viewpoint_cam_id)
                envlight_sh = self.envlight_sh_mlp(embedding_gt_image_half)
                self.envlight.set_base(envlight_sh)
                render_pkg = render(viewpoint_cam, self.gaussians, self.envlight, self.config.pipe, background, debug=False)
                # viewpoint_cam.image_width = viewpoint_cam.image_width*2
                image_half = get_half_images(img=render_pkg["render"], left=True)
                depth_half = get_half_images(img=render_pkg["depth"], left=True)
                ssim_weight = self.config.optimizer.lambda_dssim
                if mse:
                    loss = mse_loss(image_half, gt_image_half, reduction=None)
                    loss = torch.where(occluders_mask, loss, torch.nan)
                    loss = loss.nanmean()
                else:
                    loss =  l1_loss(image_half, gt_image_half, mask=occluders_mask_half)*(1 - ssim_weight) + \
                            ssim_weight *(1.0 - ssim(image_half, gt_image_half, mask=occluders_mask_half))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
