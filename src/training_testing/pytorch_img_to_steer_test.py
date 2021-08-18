from pytorch_high_level import *
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')
args = parser.parse_args()

HEIGHT = 360
WIDTH = 800
CHANNELS = 1
model_name = '/home/li/Documents/sensor_data/src/training_testing/models/steering.h5'

checkpoint = torch.load(model_name)
model = checkpoint['net']
model = model.to(device)

train_data = np.load('/home/li/Documents/sensor_data/data/0810175158/fisheyes_steering_croped_one_hot_balanced_test.npy',
                     allow_pickle=True)
print('data shape: ', train_data.shape)
X = ([i[0] for i in train_data])
Y = ([i[1] for i in train_data])
with torch.no_grad():
    y_true = []
    y_pred = []
    for j in range(0, len(X), 10):
        img = np.array(X[j], dtype=np.float32)
        time.sleep(0.001)
        img = torch.Tensor(img).view(-1, CHANNELS, HEIGHT, WIDTH).to(device)
        # now = time.time()
        ot = model(img)[0]
        ot = ot.cpu().data.numpy()
        # max_ind = np.where(np.max(output))
        y_pred.append(int(np.where(ot == np.amax(ot))[0]))
        # delta = time.time()-now
        # print(delta*1000)
        GT = Y[j]
        # max_ind = np.where(np.max(GT))
        y_true.append(int(np.where(GT == np.amax(GT))[0]))
        # print('steering angle and speed errors: ', np.fabs(GT - output))

## plot confusion matrix
classes = ['-0.34', '-0.30', '-0.25', '-0.20', '-0.15', '-0.10', '-0.05', '0.00', '0.05', '0.10', '0.15', '0.20',
           '0.25', '-0.30', '-0.34']

cm = confusion_matrix(np.array(y_true), np.array(y_pred))
# plot_confusion_matrix(cm, 'confusion_matrix.png', classes, title='confusion matrix')

df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
plt.figure(figsize=(10, 7))
# sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True) # font size
plt.show()

train_data = None
X = None
Y = None
del train_data
del X
del Y
