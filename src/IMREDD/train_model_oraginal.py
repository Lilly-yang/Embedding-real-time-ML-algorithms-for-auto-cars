import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
import rosbag
from cv_bridge import CvBridge
import cv2

IMG_HEIGHT=180
IMG_WIDTH=640

bridge = CvBridge()

topic_mux="/car/mux/ackermann_cmd_mux/output"
topic_cam="/car/camera/color/image_raw"
steering_angle=[]
#seq=[]
img_ids=[]
nb_bag=7

for j in range(1,nb_bag+1):
    nom='rec%i' %j
    output_dir="/home/julien/ros/imredd_mushr_noetic/src/follow_road/bagfiles/circuit_smplf/%s/" %nom
    folder_path='/home/julien/ros/imredd_mushr_noetic/src/follow_road/bagfiles/circuit_smplf/%s/' %nom
    bag_file='/home/julien/ros/imredd_mushr_noetic/src/follow_road/bagfiles/circuit_smplf/%s.bag' %nom
    bag = rosbag.Bag(bag_file, "r")
    #write_bag = rosbag.Bag('/home/julien/ros/imredd_mushr_noetic/src/follow_road/bagfiles/circuit_smplf/%s_ts.bag' %nom, 'w')
    
    #os.mkdir(output_dir)
    #image_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))

    # Enregistre images
    ts_camera=[]
    for topic, msg, t in bag.read_messages(topics=[topic_cam]):
        #write_bag.write(topic, msg, t=t)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        #cv2.imwrite(os.path.join(output_dir, "%04i.png" % count), cv_img)
        #img_ids.append(os.path.join(output_dir, "%04i.png" % count))
        img_ids.append(cv_img)
        ts_camera.append(t.to_nsec())
        #seq.append(msg.header.seq)
    print ("Img fait")
    #bag.close()
    
    # crée liste images

    print(len(ts_camera))
    # crée la liste steering_angle
    i=0
    for topic, msg, t in bag.read_messages(topics=[topic_mux]):
        if i<len(ts_camera):
            if i==0 :
                if t.to_nsec()>ts_camera[i]:
                    steering_angle.append(msg.drive.steering_angle)
                    #write_bag.write(topic, msg, t=t)
                    i=i+1
            if t.to_nsec()<ts_camera[i]:
                t_temp=t
                msg_temp=msg
            else : 
                steering_angle.append(msg_temp.drive.steering_angle)
                #write_bag.write(topic, msg_temp, t=t_temp)
                i=i+1
                if t.to_nsec()>ts_camera[i]:
                    steering_angle.append(msg_temp.drive.steering_angle)
                    #write_bag.write(topic, msg_temp, t=t_temp)
                    i=i+1
                t_temp=t
                msg_temp=msg
            if i== len(ts_camera)-1:
                steering_angle.append(msg.drive.steering_angle)
                #write_bag.write(topic, msg, t=t)
                
    #write_bag.close()
height, width = cv_img.shape[:2]
print(len(img_ids))

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, taille, random_hflips=False):
        #self.directory = directory
        self.taille=taille
        self.random_hflips = random_hflips
        #self.image_paths = sorted(glob.glob(os.path.join(self.directory, '*.png')))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return self.taille
    
    def __getitem__(self, idx):
        #image_path = self.image_paths[idx]
        #image = PIL.Image.open(image_path)
        image = PIL.Image.fromarray(img_ids[idx])
        #image=image.convert('L')
        image=image.crop((0,height/2,width,height))
        steering = [steering_angle[idx]]
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (IMG_HEIGHT,IMG_WIDTH))

        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, torch.tensor(steering).float()
    
dataset = XYDataset(len(img_ids), random_hflips=False)
print('dataset',len(dataset))
# Split dataset into train and test sets
test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
print('taile',len(train_dataset), len(test_dataset))


# Create data loaders to load data in batches
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

# Define Neural Network Model
#from torchsummary import summary

model = models.resnet18(weights='DEFAULT')
#print(summary(model, (3,224,224)))
model.fc = torch.nn.Linear(512,1)
device = torch.device('cuda')
model = model.to(device)

# Train regression
NUM_EPOCHS = 70
BEST_MODEL_PATH = '/home/julien/ros/imredd_mushr_noetic/src/follow_road/bagfiles/test2/best_steering_model.pth'
best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #print('lab',labels.shape)
        #print('img',images.shape)
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




# Train model du jetbot : 
#   entrée = image
#   sortie = coordonnées visées (pour en déduire l'angle de bracage)

'''folder_path='/home/julien/ros/imredd_mushr_noetic/src/follow_road/dataset/'


def get_x(path, width):
    """Gets the x value from the image filename"""
    print('x',path.split("_")[2])
    return (float(int(path.split("_")[2]) - width/2)) / (width/2)

def get_y(path, height):
    """Gets the y value from the image filename"""
    y_temp=path.split("_")[3]
    print('y',y_temp)
    return (float(height - int(y_temp.split(".")[0]))) / (height/2)

class XYDataset(torch.utils.data.Dataset):
    
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        width, height = image.size
        x = float(get_x(os.path.basename(image_path), width))
        y = float(get_y(os.path.basename(image_path), height))
        print(idx)
        print(image_path)
        print('x,y',x,y)
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, torch.tensor([x, y]).float()
    
dataset = XYDataset(folder_path, random_hflips=False)

# Split dataset into train and test sets
test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

# Create data loaders to load data in batches
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

# Define Neural Network Model

model = models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Linear(512, 2)
device = torch.device('cuda')
model = model.to(device)

# Train regression
NUM_EPOCHS = 70
BEST_MODEL_PATH = '/home/julien/ros/imredd_mushr_noetic/src/follow_road/best_steering_model_xy.pth'
best_loss = 1e9

optimizer = optim.Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        print('lab',labels.shape)
        print('img',images.shape)
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
    
    print('%f, %f' % (train_loss, test_loss))
    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss
'''
