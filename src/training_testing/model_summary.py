# import torch.utils.data
from pytorch_model import Net
# import argparse
from functions import *
# # from torchvision import models
from torchsummary import summary
#
# datasets = ['s1_p1_1', 's1_p2_1', 's1_p3_1', 's1_p4_1']
# DATA_ROOT_PATH = '/home/li/Documents/sensor_data/data/MIA'
#
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = Truet
#
# parser = argparse.ArgumentParser(description='training')
# parser.add_argument('--dataset_name', type=str, default="testing", help='suffix for the dataset name')
# args = parser.parse_args()

HEIGHT = 360 #800
WIDTH = 800 #848
CHANNELS = 1
model_name = 'models/steering_MIA.h5'

net = Net(HEIGHT, WIDTH, CHANNELS).to(device)  # define the network and send it to the gpu/cpu
print(net)
# summary(Net, (HEIGHT, WIDTH, CHANNELS))