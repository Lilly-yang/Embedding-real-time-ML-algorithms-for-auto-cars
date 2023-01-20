import os.path
import random
from functions import *


DATA_PATH = '/home/li/Documents/sensor_data/data/MIA/s1_p1_1/train'

validation_set = 0.2

train_path = []
val_path = []
for d in os.listdir(DATA_PATH):
    if os.path.isdir(os.path.join(DATA_PATH, d)):
        files = os.listdir(os.path.join(DATA_PATH, d))
        # random.shuffle(files)

        path = []
        for file in files:
            path.append([os.path.join(DATA_PATH, d, file), d])
            # x_path.append(os.path.join(train_path, d, file))
            # y.append(np.array(lable))

        spt = int(len(path) * validation_set)
        train_path += path[spt:]
        # x_train_path += x_path[spt:]
        # x_val_path += x_path[:spt]

        val_path += path[:spt]
        # y_train += y[spt:]
        # y_val += y[:spt]

    # return x_train_path, y_train, x_val_path, y_val

save_to_csv(train_path, os.path.join(DATA_PATH, 'train.csv'), type = 'w')
save_to_csv(val_path, os.path.join(DATA_PATH, 'val.csv'), type = 'w')
