from pytorch_high_level import *
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from functions import *


DATA_PATH = '/home/li/Documents/sensor_data/data/MIA/s1_p2_1/train'

parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--dataset_name', type=str, default="4", help='suffix for the dataset name')
args = parser.parse_args()

HEIGHT = 360
WIDTH = 800
CHANNELS = 1
model_name = 'models/steering_MIA.h5'

checkpoint = torch.load(model_name)
model = checkpoint['net']
model = model.to(device)

# test_data = np.load(
#     '/home/li/Documents/Embedding-real-time-ML-algorithms-for-auto-cars/data/0810175158/fisheyes_steering_2_cnnel_croped_unshuffle.npy',
#     allow_pickle=True)
# print('data shape: ', test_data.shape)
# X = ([i[0] for i in test_data])
# Y = ([i[1] for i in test_data])

x_path, y_array = read_csv(os.path.join(DATA_PATH, 'val.csv'))

TD_test = Dataset(x_path, y_array, CHANNELS, HEIGHT, WIDTH)
DL_DS_test = torch.utils.data.DataLoader(TD_test, batch_size=1, shuffle=True)

with torch.no_grad():
    y_true = []
    y_pred = []
    # x, y = DL_DS_test
    for (i, batch) in enumerate(DL_DS_test):
        # print("\nBatch = " + str(i))
        batch_X = batch['predictors']  # [3,7]
        batch_Y = batch['political']  # [3]

        ot = model(batch_X)[0]
        ot = ot.cpu().data.numpy()

        y_pred.append(int(np.where(ot == np.amax(ot))[0]))
        batch_Y = batch_Y.cpu().data.numpy()
        batch_Y = batch_Y[0]
        y_true.append(int(np.where(batch_Y == np.amax(batch_Y))[0]))

    # for i in range(len(y)):
    #     # cv2.imshow("Image window", X[j][0])
    #     # cv2.waitKey(20)
    #
    #     # img = np.array(X[j], dtype=np.float32) / .255
    #     # # time.sleep(0.001)
    #     # img = torch.Tensor(img).view(-1, CHANNELS, HEIGHT, WIDTH).to(device)
    #     # # now = time.time()
    #     ot = model(x[i])[0]
    #     ot = ot.cpu().data.numpy()
    #     # max_ind = np.where(np.max(output))
    #     y_pred.append(int(np.where(ot == np.amax(ot))[0]))
    #     # delta = time.time()-now
    #     # print(delta*1000)
    #     GT = y[i]
    #     # max_ind = np.where(np.max(GT))
    #     y_true.append(int(np.where(GT == np.amax(GT))[0]))
    #     # print('steering angle and speed errors: ', np.fabs(GT - output))
    #     # print('pred:', y_pred[-1], 'true', y_true[-1])

## plot confusion matrix
classes = ['-0.34', '-0.17', '0', '0.17', '0.34'] # ['-0.34', '-0.30', '-0.25', '-0.20', '-0.15', '-0.10', '-0.05', '0.00', '0.05', '0.10', '0.15', '0.20',
           #'0.25', '0.30', '0.34']

cm = confusion_matrix(np.array(y_true), np.array(y_pred))
# plot_confusion_matrix(cm, 'confusion_matrix.png', classes, title='confusion matrix')

df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
plt.figure(figsize=(10, 7))
# sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True)  # font size
plt.show()

test_data = None
X = None
Y = None
del test_data
del X
del Y
