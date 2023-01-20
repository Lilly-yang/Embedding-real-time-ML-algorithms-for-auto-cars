import os.path

import torch.utils.data
from pytorch_model import Net
import argparse
from functions import *

datasets = ['s1_p1_1', 's1_p2_1', 's1_p3_1', 's1_p4_1']
DATA_ROOT_PATH = '/home/li/Documents/sensor_data/data/MIA'

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--dataset_name', type=str, default="testing", help='suffix for the dataset name')
args = parser.parse_args()

HEIGHT = 360 #800
WIDTH = 800 #848
CHANNELS = 1
model_name = 'models/steering_MIA.h5'

net = Net(HEIGHT, WIDTH, CHANNELS).to(device)  # define the network and send it to the gpu/cpu
train_log = []
torch.cuda.empty_cache()

# train_data = np.load('data/0810175158/fisheyes_steering_2_cnnel_croped_unshuffle.npy', allow_pickle=True)
x_train_path, y_train = [], []
x_val_path, y_val = [], []
for dataset in datasets:
    DATA_PATH = os.path.join(DATA_ROOT_PATH, dataset, 'train')
    print('load data from: ', DATA_PATH)
    x_train_path, y_train = read_csv(os.path.join(DATA_PATH, 'train.csv'))
    x_val_path, y_val = read_csv(os.path.join(DATA_PATH, 'val.csv'))

print('traning data: ', len(x_train_path), '\n', 'val data: ', len(x_val_path))
TD_train = Dataset(x_train_path, y_train, CHANNELS, HEIGHT, WIDTH)
DL_DS_train = torch.utils.data.DataLoader(TD_train, batch_size=8, shuffle=True)

TD_val = Dataset(x_val_path, y_val, CHANNELS, HEIGHT, WIDTH)
DL_DS_val = torch.utils.data.DataLoader(TD_val, batch_size=8, shuffle=True)

# print('data shape: ', train_data.shape)
# X = torch.Tensor([i[0] for i in train_data]).view(-1, CHANNELS, HEIGHT, WIDTH)
# X = X / .255
# Y = torch.Tensor([i[1] for i in train_data])
# del train_data

# train_log = fit(net, X, Y, train_log, optimizer='adam', loss_function='mean_square', validation_set=0.2, BATCH_SIZE=8,
#                 EPOCHS=30, model_name = model_name)
train_log = fit_dataloader(net, DL_DS_train, DL_DS_val, train_log, EPOCHS=30, model_name = model_name)

a = time.localtime(time.time())
log_file = 'steering_log_{}_{}_{}_{}_{}.npy'.format(a.tm_year, a.tm_mon, a.tm_mday, a.tm_hour, a.tm_min)
np.save(log_file, np.array(train_log))

# TODO: add a snapshot step to autosave model.
