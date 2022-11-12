import torch
import torchvision
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

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
            nn.Linear(128, len(classes)),
        )

    def forward(self, x1, x2):
        x1 = self.image_features(x1)
        x = torch.cat((x1, x2), 1)
        x = self.combined_layers(x)
        x = torch.sigmoid(x)
        return x


model = Network(resnet_model).to(device)

model.load_state_dict(torch.load("model.pth"))
model.eval()

classes_to_idx = {
    "Acmispon glaber": 0,
    "Amsinckia intermedia": 1,
    "Apiastrum angustifolium": 2,
    "Calystegia macrostegia": 3,
    "Camissonia bistorta": 4,
    "Carduus pycnocephalus": 5,
    "Centaurea melitensis": 6,
    "Corethrogyne filaginifolia": 7,
    "Croton setigerus": 8,
    "Encelia farinosa": 9,
    "Ericameria pinifolia": 10,
    "Eriogonum fasciculatum": 11,
    "Eriogonum gracile": 12,
    "Erodium Botrys": 13,
    "Erodium cicutarium": 14,
    "Heterotheca grandiflora": 15,
    "Hirschfeldia incana": 16,
    "Lepidospartum squamatum": 17,
    "Lessingia glandulifera": 18,
    "Malosma laurina": 19,
    "Marah Macrocarpa": 20,
    "Mirabilis laevis": 21,
    "Olea europaea": 22,
    "Penstemon spectabilis": 23,
    "Phacelia distans": 24,
    "Rhus integrifolia": 25,
    "Ribes aureum": 26,
    "Salvia apiana": 27,
    "Sambucus nigra": 28,
    "Solanum umbelliferum": 29,
}

idx_to_classes = [
    "Acmispon glaber",
    "Amsinckia intermedia",
    "Apiastrum angustifolium",
    "Calystegia macrostegia",
    "Camissonia bistorta",
    "Carduus pycnocephalus",
    "Centaurea melitensis",
    "Corethrogyne filaginifolia",
    "Croton setigerus",
    "Encelia farinosa",
    "Ericameria pinifolia",
    "Eriogonum fasciculatum",
    "Eriogonum gracile",
    "Erodium Botrys",
    "Erodium cicutarium",
    "Heterotheca grandiflora",
    "Hirschfeldia incana",
    "Lepidospartum squamatum",
    "Lessingia glandulifera",
    "Malosma laurina",
    "Marah Macrocarpa",
    "Mirabilis laevis",
    "Olea europaea",
    "Penstemon spectabilis",
    "Phacelia distans",
    "Rhus integrifolia",
    "Ribes aureum",
    "Salvia apiana",
    "Sambucus nigra",
    "Solanum umbelliferum",
]

def classify_pollen(images):
    with torch.no_grad():
        output = model(torch.stack(images))
        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        return [idx_to_classes for idx in list(predictions.cpu().numpy())]
