import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from pprint import pprint
from .libary.xception_detector import XceptionDetector

def make_model():
    model = XceptionDetector(2)
    return model

if __name__ == "__main__":
    
    from pprint import pprint
    model = make_model()
    pprint(model)
