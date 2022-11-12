import torch
import torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from api.utils import idx_to_classes, classes_to_idx, path_to_models

device = "cuda" if torch.cuda.is_available() else "cpu"

image_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


resnet_model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
).to(device)
resnet_model.fc = Identity()
# output = resnet_model(x) # Size: (1, 2048)
for param in resnet_model.parameters():
    param.requires_grad = False
resnet_model.eval()


class Network(nn.Module):
    def __init__(self, image_features):
        super().__init__()

        self.image_features = image_features

        # TODO: Research if there are better fc layer setups
        self.combined_layers = nn.Sequential(
            nn.Linear(
                2048, 1024
            ),  # the number of neurons in the first layer should be 2048 (# of resnet features)
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, len(idx_to_classes)),
        )

    def forward(self, x):
        x = self.image_features(x)
        x = self.combined_layers(x)
        return x


model = Network(resnet_model).to(device)

model.load_state_dict(torch.load(str(path_to_models / "resnet50.final.pth"), map_location=device))
model.eval()

def classify(images):
    converted_images = [image_transforms(Image.fromarray(img)) for img in images]
    with torch.no_grad():
        output = model(torch.stack(converted_images))
        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        return [idx_to_classes[idx] for idx in list(predictions.cpu().numpy())]
