import csv
import shutil
import matplotlib.pyplot as plt
import decimal
import torch as T
from training_testing.pytorch_high_level import *


def mk_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def plot_chart(Y, X=None, title='Line Chart', xlable = 'X', ylable = 'Y'):
    if X == None:
        plt.plot(Y)
    else:
        plt.plot(Y, X)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.show()


# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


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


class Dataset(T.utils.data.Dataset):

    def __init__(self, x_tmp, y_tmp, CHANNELS, HEIGHT, WIDTH):
        # x_tmp = np.loadtxt(src_file, max_rows=num_rows,
        #   usecols=range(0,7), delimiter="\t", skiprows=0,
        #   dtype=np.float32)
        # y_tmp = np.loadtxt(src_file, max_rows=num_rows,
        #   usecols=7, delimiter="\t", skiprows=0, dtype=np.long)

        self.x_data = x_tmp  # T.tensor(x_tmp, dtype=T.float32).to(device)
        # self.y_data = T.tensor(y_tmp, dtype=T.float32).view(-1,1,5).to(device)
        self.y_data = T.tensor(y_tmp, dtype=T.float32).to(device)
        self.CHANNELS = CHANNELS
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH

    def __len__(self):
        return len(self.x_data)  # required

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        # Load data and get label
        preds = cv2.imread(self.x_data[idx], cv2.IMREAD_GRAYSCALE)
        # preds /= .255
        preds = T.tensor(preds, dtype=T.float32).view(self.CHANNELS,self.HEIGHT,self.WIDTH).to(device)
        # preds = T.tensor(preds, dtype=T.float32).to(device)
        pol = self.y_data[idx]

        sample = \
            {'predictors': preds, 'political': pol}

        return sample


def read_dataloader(DL_DS):
    for (batch_idx, batch) in enumerate(DL_DS):
        print("\nBatch = " + str(batch_idx))
        X = batch['predictors']  # [3,7]
        Y = batch['political']  # [3]
        # print(X)
        # print(Y)

    return X,Y


def save_to_txt(l, name = 'demo.txt'):
    with open(name, 'w') as f:
        for line in l:
            f.write(line)
            f.write('\n')

def save_to_csv(l, name = 'demo.csv', type = 'w'):
    with open(name, type) as f:
        for line in l:
            write = csv.writer(f)
            write.writerow(line)

    print('saved: ', name)

def read_csv(path):
    print('read: ', path)
    x = []
    y = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            x.append(row[0])
            y.append(onehot(int(row[1]), 5))

    return x,y

def onehot(data, lenth):
    lable = [0] * lenth
    lable[data] = 1

    return np.array(lable)