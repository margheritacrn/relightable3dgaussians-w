from os import makedirs
import torch
import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser
import hydra
from train import training
import optuna
import gc
import numpy as np
from functools import partial
from eval_with_gt_envmaps import render_and_evaluate_tuning_scenes
from tensorboard.plugins.hparams import api as hp
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from numba import cuda


scenes = ["schloss"]


def objective(trial: optuna.trial.Trial, output_path: str, nerfosr_path: str):

    psnrs = []
    
    #torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cuda.select_device(0)
    torch.cuda.empty_cache()

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    for scene in scenes:
        # Initialize the base config
        with hydra.initialize(config_path="configs"):
            cfg = hydra.compose(config_name="relightable3DG-W")
    
        model_path = output_path + "/" + scene
        source_path = nerfosr_path + "/" + scene

        cfg.dataset.source_path = source_path
        cfg.dataset.model_path = model_path
        cfg.optimizer.iterations = 1_000 #40_000
        cfg.dataset.logger = False
        cfg.dataset.eval = False

        # Update cfg with Optuna-suggested hyperparameters
        cfg.init_sh_mlp = False
        cfg.init_embeddings = False
        cfg.optimizer.lambda_sky_gauss = trial.suggest_categorical("lambda_sky_gauss", [0.05, 0.5]) # 0 
        cfg.optimizer.lambda_envlight = trial.suggest_categorical("lambda_envlight", [1, 10, 100])  # 2
        cfg.optimizer.reg_normal_from_iter = trial.suggest_categorical("reg_normal_from_iter", [2_000, 3_000, 4_000, 5_000]) # 0
        cfg.embeddings_dim = trial.suggest_categorical("appearance_embeddings_dim", [16, 32, 64]) # 1
        # Train
        testing_iterations = []
        saving_iterations = [1, cfg.optimizer.iterations]
        training(cfg, testing_iterations, saving_iterations)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Render and eval all with gt envmaps
        cfg.dataset.eval = True
        psnr = render_and_evaluate_tuning_scenes(cfg)
        psnrs.append(psnr)

        del cfg
        del psnr
        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        cuda.select_device(0)  # Reset the GPU device
        torch.cuda.empty_cache()
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    return np.mean(np.array(psnrs))


if __name__ == "__main__":
    parser = ArgumentParser(description="Hyperparameters tuning")
    parser.add_argument("--nerfosr_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args(sys.argv[1:])

    safe_state(True)
    

    makedirs(args.output_path, exist_ok=True)

    database_path = args.output_path + "/HP_tuning" + '.db'
    study_storage_db = 'sqlite:///' + database_path

    # Create opttuna study
    objective = partial(objective, output_path=args.output_path, nerfosr_path=args.nerfosr_path)
    study = optuna.create_study(direction="maximize", storage=study_storage_db)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # save visualizations of the results
    parall_plot = plot_parallel_coordinate(study)
    parall_plot.write_image(args.output_path + "/parall_coord.png")

    optimization_hist_plot = plot_optimization_history(study)
    optimization_hist_plot.write_image(args.output_path + "/optimization_history.png")

    param_importances_plot = plot_param_importances(study)
    param_importances_plot.write_image(args.output_path + "/param_importances.png")

    # save results
    with open(args.output_path + "/best_hyperparameters.txt", "w") as f:
        f.write("Best hyperparameters:\n")
        f.write(str(study.best_params) + "\n")
        f.write("Best value:\n")
        f.write(str(study.best_value) + "\n")
    

    print("Best trial found:")
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)

