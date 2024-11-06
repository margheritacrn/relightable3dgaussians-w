from PIL import Image
import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
import os
import numpy as np
import PIL
# TODO: allow for handling PhotoTurism Dataset as well: I think I can just change the name of the class

class NerfOSRDataset(torch.utils.data.Dataset):
    """Customized pytorch Dataset for NeRF-OSR in the wild dataset. 
    """
    def __init__(self, data_path: str, transforms = transforms.ToTensor()):
            self.path = data_path
            self.transforms = transforms
            self.images = []

            for img in os.listdir(self.path):
                self.images.append(img)


    def __getitem__(self, idx: int):
        """
        Loads and returns a single sample (image) at a given index (idx)
        """
        assert idx < len(self.images), "idx out of bounds"
            
        # load image
        img_filename = self.images[idx]
        img_path = os.path.join(self.path, img_filename)
        img = PIL.Image.open(img_path)
        # apply transformations
        item = self.transforms(img)

        return item


    def __len__(self):
        """
        Returns the number of files
        """
        return len(self.images)


def load_train_test(datapath: str, resize_dim: int = 256, batch_size: int =1, shuffle: bool =True, num_workers: int =0):
    transforms_train = transforms.Compose([transforms.Resize((resize_dim, resize_dim)), transforms.ToTensor(),
                                           transforms.GaussianNoise(mean=0.0, sigma= 0.1, clip=True)])
    transforms_test = transforms.Compose([transforms.Resize((resize_dim, resize_dim)), transforms.ToTensor()])
    data_train = NerfOSRDataset(datapath+"/train/rgb", transforms = transforms_train)
    data_test = NerfOSRDataset(datapath+"/test/rgb", transforms = transforms_test)

    trainloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    train_iter = iter(trainloader)
    test_iter = iter(testloader)

    return train_iter, test_iter, transforms_test