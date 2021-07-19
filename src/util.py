import torch

def get_device():
    if torch.cuda.is_available():
        print("GPU Available")
    else:
        print("Running on CPU")
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    return device

def get_config(path="./config.yaml"):
    with open(path, 'r') as f:
        try:
            config = Box(yaml.safe_load(f))
        except yaml.YAMLError as exc:
            print(exc)
    return config