from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

TRANSFORM = transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

CLASSES = {
        0: "FAKE",
        1: "REAL"
}