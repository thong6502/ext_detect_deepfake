from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

TRANSFORM = transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASSES = {
        0: "FAKE",
        1: "REAL"
}