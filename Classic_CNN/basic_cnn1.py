# %%
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
from pandas.core.apply import Axis
from pandas.core.internals.managers import _preprocess_slice_or_indexer
from tqdm.auto import tqdm

# %%
dataset_train = load_dataset(
    "uoft-cs/cifar10",
    split="train",  # training dataset
    # trust_remote_code=True,
    # ignore_verifications=True,  # set to True if seeing splits Error
)
dataset_train


# %%
dataset_val = load_dataset("uoft-cs/cifar10", split="test")
print(dataset_val)

# %%
print(set(dataset_train["label"]))
print(dataset_train[0]["img"])
print(dataset_train[0]["img"].size)  # gotta convert all to 32, 32
print(dataset_train[0]["img"].mode)  # gotta convert all to RGB channels


# %%
preprocess = transforms.Compose(  # same process for training + validation data
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # gives [0, 1] values
    ]
)


# %%
inputs_train = []
for record in tqdm(dataset_train):
    img = record["img"]
    lbl = record["label"]
    # print(lbl)
    if img.mode == "L":
        # print("one")
        img = img.convert("RGB")
    tensor = preprocess(img)
    inputs_train.append([tensor, lbl])
print(len(inputs_train))
print(inputs_train[0][0].shape)
print(inputs_train[0])


# %%
inputs_val = []
for record in tqdm(dataset_val):
    img = record["img"]
    lbl = record["label"]
    # print(lbl)
    if img.mode == "L":
        # print("one")
        img = img.convert("RGB")
    tensor = preprocess(img)
    inputs_val.append([tensor, lbl])
print(inputs_val[0][0].shape)
print(len(inputs_val))
print(inputs_val[0])


# %%
def get_mean_and_std(sampels_list):
    np.random.seed(0)
    # getting a random sample
    idx = np.random.randint(0, len(sampels_list), 2048)  # 2048 indexes from 5k
    # print(idx)
    tensor = torch.concat([inputs_train[i][0] for i in idx], axis=1)
    # print(tensor.shape) #for each channel, 65k pixel line x 32 columns
    tensor = tensor.swapaxes(0, 1).reshape(3, -1).T
    # print(tensor.shape)
    # print(tensor)# each column is a channel, so get its mean + std
    mean = torch.mean(tensor, axis=0)
    std = torch.std(tensor, axis=0)
    # print(mean)
    # print(std)
    del tensor
    return mean, std


mean, std = get_mean_and_std(inputs_train)
print(mean)
print(std)
# mean_v, std_v = get_mean_and_std(inputs_val)
# print(mean_v)
# print(std_v)


# %%
preprocess = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
# preprocess_val = transforms.Compose([transforms.Normalize(mean=mean_v, std=std_v)])


# %%
for i in tqdm(range(len(inputs_train))):
    tensor = preprocess(inputs_train[i][0])
    inputs_train[i][0] = tensor
inputs_train[0][0]


# %%
for i in tqdm(range(len(inputs_val))):
    tensor = preprocess(inputs_val[i][0])
    inputs_val[i][0] = tensor
inputs_val[0][0]


# %%
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    inputs_train, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    inputs_val, batch_size=batch_size, shuffle=True
)

# %%
for batch_images, batch_labels in val_loader:
    print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")
    break
for batch_images, batch_labels in train_loader:
    print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")
    break

# %%
# Data is ready, now contruct CNN architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# %%
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        # 1st; basic paterns, lines/edges...
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 2nd: basic shapes, corners...
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=4, padding=1
        )
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 3rd; more complex shapes, squares, circles...
        # + beginnig of deep stacking conv layers before pooling
        self.conv3 = nn.Conv2d(
            in_channels=192, out_channels=384, kernel_size=3, padding=1
        )
        self.relu3 = nn.ReLU()
        # 4th: recognizing component of objects, like wheels...
        # + NN plateus after reaching max channels
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, padding=1
        )
        self.relu4 = nn.ReLU()
        # 5th; recognizing full objects based on what components theyre made of
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # now smooth gradual reduction 1024-512-256-10
        # with Dropout to avoid overfitting
        self.dopout6 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(1024, 512)
        self.relu6 = nn.ReLU()
        self.dopout7 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(256, num_classes)


# %%

# %%
