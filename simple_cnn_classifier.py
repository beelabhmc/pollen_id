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
model_name = "resnet50_with_context"

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = 0  # os.cpu_count()
# %%
image_res = 224
train_transform = transforms.Compose(
    [
        transforms.Resize((image_res, image_res)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((image_res, image_res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
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

        # [image size]
        contextual_features = torch.tensor([img.size[0] / image_res])

        if self.transform:
            return self.transform(img), contextual_features, class_idx
        else:
            return img, contextual_features, class_idx


# %%
train_set = PollenDataset(
    pollen_grains_dir / "train", min_num=10, transform=train_transform
)
test_set = PollenDataset(
    pollen_grains_dir / "test", classes=train_set.classes, transform=test_transform
)

classes = train_set.classes

# Print how many of each class we have
print(
    dict(
        zip(
            train_set.classes,
            dict(Counter(train_set.targets + test_set.targets)).values(),
        )
    )
)

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
    img = img / 4 + 0.5  # unnormalize
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get random training images with iter function
dataiter = iter(train_loader)
images, context, labels = dataiter.next()

# call function on our images
imshow(torchvision.utils.make_grid(images))

# print the class of the image
print(" ".join("%s" % classes[labels[j]] for j in range(BATCH_SIZE)))
# %%
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
            nn.Linear(2048 + 1, 1024), # the number of neurons in the first layer should be 2048 (# of resnet features) + (# of context features)
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

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
optimizer = torch.optim.RAdam(params=model.parameters(), lr=0.001)
# %%
metric = torchmetrics.Accuracy(num_classes=len(classes)).to(device)

# From: https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


train_loss = []
train_acc = []
test_loss = []
test_acc = []

early_stopper = EarlyStopper(patience=3, min_delta=0.2)

for epoch in range(250):
    running_loss = 0
    running_acc = 0
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, context, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            images, context, target = images.to(device), context.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(images, context)
            loss = criterion(output, target)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            metric.reset()
            accuracy = metric(predictions, target)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100.0 * accuracy.item())
            running_loss += loss.item()
            running_acc += accuracy.item()
    torch.save(
        model.state_dict(), model_save_dir / f"{model_name}.snapshot-epoch{epoch}.pth"
    )

    train_loss.append(running_loss / len(train_loader))
    train_acc.append(running_acc / len(train_loader))

    running_loss = 0
    running_acc = 0
    model.eval()
    with torch.no_grad():
        for images, context, target in test_loader:
            images, context, target = images.to(device), context.to(device), target.to(device)
            output = model(images, context)
            loss = criterion(output, target)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            metric.reset()
            accuracy = metric(predictions, target)

            running_loss += loss.item()
            running_acc += accuracy.item()

    test_loss.append(running_loss / len(test_loader))
    test_acc.append(running_acc / len(test_loader))

    if early_stopper.early_stop(test_loss[-1]):
        break

print(f"Finished Training")
torch.save(model.state_dict(), model_save_dir / f"{model_name}.final.pth")
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
        images, context, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        output = model(images, context)
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
# plt.xlabel("predicted")
# plt.ylabel("true label")
ax = sn.heatmap(df_cm, annot=True)
ax.set(xlabel='predicted', ylabel='true')
plt.show()
# %%
m = precision_recall_fscore_support(
    np.array(combined_labels), np.array(combined_predictions)
)
print(f"precision: {m[0]} \n")
print(f"recall: {m[1]} \n")
print(f"f1 score: {m[2]} \n")
# %%
