import numpy as np

config = {
    "01-09_14_00_IMG_0706": {
        "mask_path": "../data/nerfosr_colmap/st/test/mask/01-09_14_00_IMG_0706.png",
        "env_map_path": "../data/nerfosr_colmap/st/test/ENV_MAP_CC/01-09_14_00/20210901_140700.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-5/3),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 30,
        },
        "sun_angles": [-0.3 * np.pi, 0.1 * np.pi],
    },
    "24-08_11_30_IMG_9690": {
        "mask_path": "../data/nerfosr_colmap/st/test/mask/24-08_11_30_IMG_9690.png",
        "env_map_path": "../data/nerfosr_colmap/st/test/ENV_MAP_CC/24-08_11_30/20210824_114352.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-5/3),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 30,
        },
        "sun_angles": [0.9 * np.pi, 1.3 * np.pi],
    },
    "24-08_16_30_IMG_0061": {
        "mask_path": "../data/nerfosr_colmap/st/test/mask/24-08_16_30_IMG_0061.png",
        "env_map_path": "../data/nerfosr_colmap/st/test/ENV_MAP_CC/24-08_16_30/20210824_173357.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-5/3),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 30,
        },
        "sun_angles": [0.1 * np.pi, 0.5 * np.pi],
    },
    "25-08_19_30_IMG_0306": {
        "mask_path": "../data/nerfosr_colmap/st/test/mask/25-08_19_30_IMG_0306.png",
        "env_map_path": "../data/nerfosr_colmap/st/test/ENV_MAP_CC/25-08_19_30/20210825_185546.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-5/3),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 30,
        },
        "sun_angles": [0.0 * np.pi, 0.3 * np.pi],
    },
    "31-08_07_30_IMG_0501": {
        "mask_path": "../data/nerfosr_colmap/st/test/mask/31-08_07_30_IMG_0501.png",
        "env_map_path": "../data/nerfosr_colmap/st/test/ENV_MAP_CC/31-08_07_30/20210831_075634.jpg",
        "initial_env_map_rotation": {
            "x": np.pi*(-5/3),
            "y": 0,
            "z": 0,
        },
        "env_map_scaling": {
            "threshold": 0.99,
            "scale": 30,
        },
        "sun_angles": [-0.8 * np.pi, -0.3 * np.pi],
    },
}
