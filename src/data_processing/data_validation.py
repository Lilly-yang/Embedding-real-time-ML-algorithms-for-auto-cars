import numpy as np
import cv2

data_path = '/home/li/Documents/Embedding-real-time-ML-algorithms-for-auto-cars/data/0810175158/fisheyes_steering_2_cnnel_croped_unshuffle.npy'
train_data = np.load(data_path, allow_pickle=True)

# for i in range(10):
for i in range(len(train_data)):
    cv2.imshow("Image window", train_data[i][0][0])
    cv2.waitKey(20)

    GT = train_data[i][1]
    print(int(np.where(GT == np.amax(GT))[0]))
