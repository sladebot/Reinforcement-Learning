import torch


def get_device():
    if torch.cuda.is_available():
        print("GPU Available")
    else:
        print("Running on CPU")
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    return device
