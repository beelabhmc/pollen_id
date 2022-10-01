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
from collections import Counter

# %%
pollen_grains_dir = pathlib.Path("pollen_grains")
model_save_dir = pathlib.Path("models")
model_save_dir.mkdir(parents=True, exist_ok=True)


torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = 0  # os.cpu_count()
# %%
image_res = 256
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
    def __init__(self, target_dir, min_num=0, classes=[], transform=None):
        super().__init__()
        self.classes = []
        self.paths = []
        all_classes = sorted(
            [dir.name for dir in os.scandir(target_dir) if dir.is_dir()]
        )
        for c in all_classes:
            imgs = [
                f
                for f in (pathlib.Path(target_dir) / c).glob("**/*.*")
                if f.suffix.lower() in utils.img_suffixes
            ]
            if len(imgs) >= min_num and ((classes and c in classes) or not classes):
                self.classes.append(c)
                self.paths.extend(imgs)

        self.transform = transform
        self.class_to_idx = {classname: i for i, classname in enumerate(self.classes)}

        self.targets = [
            self.class_to_idx[self.get_class_from_path(p)] for p in self.paths
        ]

    def load_image(self, idx):
        return Image.open(self.paths[idx])

    def get_class_from_path(self, path):
        return path.parents[2].name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        class_name = self.get_class_from_path(self.paths[idx])
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


# %%
train_set = PollenDataset(
    pollen_grains_dir / "train", min_num=50, transform=train_transform
)
test_set = PollenDataset(
    pollen_grains_dir / "test", classes=train_set.classes, transform=test_transform
)

classes = train_set.classes

# Print how many of each class we have
print(dict(zip(train_set.classes, dict(Counter(train_set.targets + test_set.targets)).values())))

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
print(" ".join("%s" % classes[labels[j]] for j in range(BATCH_SIZE)))
# %%
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, len(classes))
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
# %%
metric = torchmetrics.Accuracy().to(device)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(50):
    running_loss = 0
    running_acc = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            accuracy = metric(predictions, target)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100.0 * accuracy.item())
            running_loss += loss.item()
            running_acc += accuracy.item()
    torch.save(model.state_dict(), model_save_dir / f"resnet50.snapshot-epoch{epoch}.pth")

    train_loss.append(running_loss / len(train_loader))
    train_acc.append(running_acc / len(train_loader))
    
    
    running_loss = 0
    running_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            accuracy = metric(predictions, target)

            running_loss += loss.item()
            running_acc += accuracy.item()
    
    test_loss.append(running_loss / len(test_loader))
    test_acc.append(running_acc / len(test_loader))

print(f"Finished Training")
torch.save(model.state_dict(), model_save_dir / "resnet50.final.pth")
# %%
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label="train")
plt.plot(test_loss, label="test")
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc)
plt.plot(test_acc)
plt.title("Accuracy")

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.show()
# %%
combined_labels = []
combined_predictions = []

model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        output = model(images)
        predictions = output.argmax(dim=1, keepdim=True).squeeze()

        labels = labels.data.cpu().numpy()
        combined_labels.extend(labels)
        combined_predictions.extend(list(predictions.cpu().numpy()))

# %%
metric.reset()
accuracy = metric(torch.tensor(combined_predictions), torch.tensor(combined_labels))
print("accuracy", accuracy.item())
# %%
# Confusion Matrix
cf_matrix = confusion_matrix(combined_labels, combined_predictions)
df_cm = pd.DataFrame(
    cf_matrix,
    index=[i for i in classes],
    columns=[i for i in classes],
)
plt.figure(figsize=(12, 7))
plt.xlabel("predicted")
plt.xlabel("true label")
sn.heatmap(df_cm, annot=True)
# %%
m = precision_recall_fscore_support(
    np.array(combined_labels), np.array(combined_predictions)
)
print(f"precision: {m[0]} \n")
print(f"recall: {m[1]} \n")
print(f"f1 score: {m[2]} \n")
# %%
