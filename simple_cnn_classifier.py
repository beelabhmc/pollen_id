# %%
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

import pathlib
import os
import utils
from PIL import Image
import torchvision.transforms as transforms

from tqdm.autonotebook import tqdm
# %%
pollen_grains_dir = pathlib.Path("pollen_grains")

torch.manual_seed(42) 
torch.cuda.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_WORKERS = 0 # os.cpu_count()
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
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

# get random training images with iter function
dataiter = iter(train_loader)
images, labels = dataiter.next()

# call function on our images
imshow(torchvision.utils.make_grid(images))

# print the class of the image
print(' '.join('%s' % full_pollen_dataset.classes[labels[j]] for j in range(BATCH_SIZE)))

# %%
class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
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


net = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(full_pollen_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
# %%
for epoch in range(10):  # loop over the dataset multiple times
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            correct = (predictions == target).sum().item()
            accuracy = correct / BATCH_SIZE
            
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

print('Finished Training')
# %%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in full_pollen_dataset.classes}
total_pred = {classname: 0 for classname in full_pollen_dataset.classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[full_pollen_dataset.classes[label]] += 1
            total_pred[full_pollen_dataset.classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
# %%
