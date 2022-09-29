# %%
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import pathlib
import os
import utils
from PIL import Image
import torchvision.transforms as transforms

from tqdm.autonotebook import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import seaborn as sn

# %%
pollen_grains_dir = pathlib.Path("pollen_grains")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = 0  # os.cpu_count()
# %%
image_res = 64
train_transform = transforms.Compose(
    [
        transforms.Resize((image_res, image_res)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # mean = 0.5, std = 0.5
    ]
)
test_transform = transforms.Compose(
    [transforms.Resize((image_res, image_res)), transforms.ToTensor()]
)
# %%
# Loosely based on: https://www.learnpytorch.io/04_pytorch_custom_datasets/
class PollenDataset(torch.utils.data.Dataset):
    def __init__(self, target_dir, transform=None):
        self.paths = [
            f
            for f in pathlib.Path(target_dir).glob("*/*.*")
            if f.suffix.lower() in utils.img_suffixes
        ]
        self.transform = transform
        self.classes = sorted([dir.name for dir in os.scandir(target_dir)])
        self.class_to_idx = {classname: i for i, classname in enumerate(self.classes)}

    def load_image(self, idx):
        return Image.open(self.paths[idx])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


# %%
full_pollen_dataset = PollenDataset(pollen_grains_dir)

train_set_size = int(len(full_pollen_dataset) * 0.8)
test_set_size = len(full_pollen_dataset) - train_set_size

train_set, test_set = torch.utils.data.random_split(
    full_pollen_dataset, [train_set_size, test_set_size]
)

train_set.dataset.transform = train_transform
test_set.dataset.transform = test_transform
# %%
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
)


def imshow(img):
    """function to show image"""
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get random training images with iter function
dataiter = iter(train_loader)
images, labels = dataiter.next()

# call function on our images
imshow(torchvision.utils.make_grid(images))

# print the class of the image
print(
    " ".join("%s" % full_pollen_dataset.classes[labels[j]] for j in range(BATCH_SIZE))
)

# %%
class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,  # how big is the square that's going over the image?
                stride=1,  # default
                padding=1,
            ),  # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2, stride=2
            ),  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion


model = TinyVGG(
    input_shape=3,  # number of color channels (3 for RGB)
    hidden_units=10,
    output_shape=len(full_pollen_dataset.classes),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
# %%
metric = torchmetrics.Accuracy().to(device)

for epoch in range(50):  # loop over the dataset multiple times
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            accuracy = metric(predictions, target)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100.0 * accuracy.item())

# accuracy = metric.compute()
print(f"Finished Training")
# %%
combined_labels = []
combined_predictions = []

model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        output = model(images)
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()

        labels = labels.data.cpu().numpy()
        combined_labels.extend(labels)
        combined_predictions.extend(output)

# %%
metric.reset()
accuracy = metric(torch.tensor(combined_predictions), torch.tensor(combined_labels))
print("accuracy", accuracy.item())
# %%
# Confusion Matrix
cf_matrix = confusion_matrix(combined_labels, combined_predictions)
df_cm = pd.DataFrame(
    cf_matrix,
    index=[i for i in full_pollen_dataset.classes],
    columns=[i for i in full_pollen_dataset.classes],
)
plt.figure(figsize=(12, 7))
plt.xlabel('predicted')
plt.xlabel('true label')
sn.heatmap(df_cm, annot=True)
# %%
m = precision_recall_fscore_support(np.array(combined_labels), np.array(combined_predictions))
print(f"precision: {m[0]} \n")
print(f"recall: {m[1]} \n")
print(f"f1 score: {m[2]} \n")
# %%
