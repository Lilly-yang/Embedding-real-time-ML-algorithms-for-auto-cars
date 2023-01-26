import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image
import os
import numpy as np
import cv2


IMG_HEIGHT=360
IMG_WIDTH=800

steering_angle=[]
img_ids=[]

DATA_PATH = '/home/li/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/balanced'

onlydirs = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

final_out = []
for ind, dir in enumerate(onlydirs):
    # load data directly from rosbag
    CA_PATH = os.path.join(DATA_PATH, dir)

    final_out += [dir + '/' + f for f in os.listdir(CA_PATH) if os.path.isfile(os.path.join(CA_PATH, f))]

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, taille, random_hflips=False):
        self.taille=taille
        self.random_hflips = random_hflips
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return self.taille
    
    def __getitem__(self, idx):
        # image = PIL.Image.fromarray(img_ids[idx])
        image = cv2.imread(os.path.join(DATA_PATH, final_out[idx]))
        image = PIL.Image.fromarray(image)
        # image=image.crop((0,height/2,width,height))
        # steering = [steering_angle[idx]]
        steering = float(final_out[idx].split('_')[1])
        # image = self.color_jitter(image)
        image = transforms.functional.resize(image, (IMG_HEIGHT,IMG_WIDTH))

        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, torch.tensor(steering).float()
    
dataset = XYDataset(len(final_out), random_hflips=True)
print('dataset',len(dataset))

test_percent = 0.2
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
print('taile',len(train_dataset), len(test_dataset))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

model = models.resnet18(True)
model.fc = torch.nn.Linear(512,1)
device = torch.device('cpu') #cuda
model = model.to(device)

# Train regression
NUM_EPOCHS = 70
BEST_MODEL_PATH = '/home/li/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/src/IMREDD/best_steering_model.pth'
best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += float(loss)
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    
    model.eval()
    test_loss = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        test_loss += float(loss)
    test_loss /= len(test_loader)
    
    print('%f, %f, %i' % (train_loss, test_loss, epoch))
    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss
