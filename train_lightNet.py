from scene.light_model import lightNet
from data.dataloader import load_data
#TODO use batch training, number of epochs, learning rate, optimizer (here or inside lightNet (?))

def train_lightNet(data_path: str, num_epochs: int, resize_dim: int, batch_size: int):
    pass
    """model = lightNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_iter, test_iter = load_data(data_path, resize_dim=resize_dim, batch_size=batch_size)
    """