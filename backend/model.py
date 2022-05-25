import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_model(model_path: str = "../../assets/mask_task/model.pth") -> MyEfficientNet:
    """Model을 가져옵니다"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyEfficientNet(num_classes=18).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
