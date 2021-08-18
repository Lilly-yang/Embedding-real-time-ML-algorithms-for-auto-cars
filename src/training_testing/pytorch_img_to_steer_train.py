from pytorch_high_level import *
from pytorch_model import Net

# from keras.utils.data_utils import Sequence
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.tensorflow import balanced_batch_generator

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import argparse


parser = argparse.ArgumentParser(description='training')
parser.add_argument('--dataset_name', type=str, default="testing", help='suffix for the dataset name')
args = parser.parse_args()

HEIGHT = 360 #800
WIDTH = 800 #848
CHANNELS = 1
model_name = 'src/training_testing/models/steering.h5'

net = Net(HEIGHT, WIDTH, CHANNELS).to(device)  # define the network and send it to the gpu/cpu
# net = checkpoint['net']
train_log = []
torch.cuda.empty_cache()
train_data = np.load('/home/li/Documents/sensor_data/data/0810175158/fisheyes_steering_croped_one_hot_balanced_mini.npy', allow_pickle=True)
print('data shape: ', train_data.shape)
# np.random.shuffle(train_data)
X = torch.Tensor([i[0] for i in train_data]).view(-1, CHANNELS, HEIGHT, WIDTH)
X = X / .255
Y = torch.Tensor([i[1] for i in train_data])
del train_data

train_log = fit(net, X, Y, train_log, optimizer='adam', loss_function='mean_square', validation_set=0.2, BATCH_SIZE=8,
                EPOCHS=30, model_name = model_name)
# state = {'net': net}
# torch.save(state, model_name, _use_new_zipfile_serialization=False)
loss_function = None
optimizer = None
X = None
Y = None
del X
del Y
del loss_function
del optimizer

a = time.localtime(time.time())
log_file = 'steering_log_{}_{}_{}_{}_{}.npy'.format(a.tm_year, a.tm_mon, a.tm_mday, a.tm_hour, a.tm_min)
np.save(log_file, np.array(train_log))
# TODO: add a snapshot step to autosave model.
