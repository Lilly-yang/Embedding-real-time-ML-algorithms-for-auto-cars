import cv2
import random
from scipy import ndimage
import numpy as np
import math
import os
import shutil

import rosbag
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()  # ros img transfer to cv2


class transfer_to_one_hot():
    def __init__(self, rng, step):
        self.oh_n = int(math.ceil((rng[1] - rng[0]) / step) + 1)
        print('one hot size is', self.oh_n)
        self.step = step
        self.count = [0] * self.oh_n

    def transter(self, data):
        one_hot = [0] * self.oh_n
        ind = int(round(data / self.step + math.floor(self.oh_n / 2)))
        try:
            one_hot[ind] = 1
            # print(data, ind, one_hot)
            self.count[ind] += 1
        except:
            print('---WORNING---', data, ind, one_hot)
            exit()

        return one_hot


def balance(count, final_out, model='ave'):
    print('------data balancing------')
    print('oranginal ration: ', count)

    if model == 'min':
        num = min(count)
        print('balance based on the minimum ration: ', num)
    elif model == 'max':
        num = max(count)
        print('balance based on the maximum ration: ', num)
    else:
        num = sum(count) // len(count)
        print('balance based on the average ration: ', num)

    new_count = [[]] * len(count)
    for ind, class_num in enumerate(count):
        print('---the %d class---' % (ind + 1))
        tmp = []

        if class_num >= num:
            resultList = random.sample(range(0, class_num), num)
            for i in resultList:
                tmp.append(final_out[ind][i])
        elif class_num == 0:
            continue
        else:
            tim = num // class_num
            remainder = num % class_num
            # print('will add %d times sata and %d extra data'%(tim, remainder))

            for i in range(tim):
                tmp += final_out[ind]

            if remainder:
                resultList = random.sample(range(0, class_num), remainder)
                for i in resultList:
                    tmp.append(final_out[ind][i])


            random.shuffle(tmp)

        final_out[ind] = tmp
        new_count[ind] = len(final_out[ind])

    print('balanced finished: ', new_count)

    return final_out


def augmentation(img):
    # img = img.rotate(random.randint(0,15))
    img = ndimage.rotate(img, random.randint(-15, 15), reshape=False)

    cord = [random.randint(180, 370), random.randint(0, 24)]

    img = img[cord[0]:cord[0] + 360, cord[1]:cord[1] + 800]  # stanard = [330:690, 24:824]

    return img


def preprocess(data, save_path, bag_file = 'bag'):
    for ind, d in enumerate(data):
        print('------%d / %d------'%(ind, len(data)))

        new_dir = os.path.join(save_path, str(ind))
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        for i, msg in enumerate(d):
            idxImg = int(msg[0])

        ## image load
            camera_image = bridge.imgmsg_to_cv2(camera[idxImg][1], "bgr8")

        # # prepocessing
        # fisheye1_image = cv2.cvtColor(fisheye1_image, cv2.COLOR_BGR2GRAY)
            # camera_image = augmentation(camera_image)

        # cv2.imshow('image', fisheye1_image)
        # cv2.waitKey(0)

            cv2.imwrite(new_dir+'/'+os.path.splitext(bag_file)[0]+'_'+str(msg[1])[:5]+'_'+str(msg[2])[:5]+'_'+str(i)+'.jpg', camera_image)
            # print('save to:', new_dir+'/'+str(idxImg).zfill(5)+'camera'+'.jpg')


if __name__ == "__main__":
    PATH_TO_INPUT_BAG = '/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/data'
    PATH_TO_SAVE = '/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/processed'
    topics = ['/car/mux/ackermann_cmd_mux/output',
              '/car/camera/color/image_raw',]

    onlyfiles = [f for f in os.listdir(PATH_TO_INPUT_BAG) if os.path.isfile(os.path.join(PATH_TO_INPUT_BAG, f))]

    for bag_file in onlyfiles:
        # load data directly from rosbag
        BAG_PATH = os.path.join(PATH_TO_INPUT_BAG, bag_file)
        print('---read bag: ', BAG_PATH)
        bag = rosbag.Bag(BAG_PATH)

        camera = []
        steering_angle = []
        speed = []

        print('read message', topics)
        for topic, msg, t in bag.read_messages(topics=topics):
            if topic == '/car/camera/color/image_raw':  # 30/s
                camera.append((t, msg))
            elif topic == '/car/mux/ackermann_cmd_mux/output':  # 63/s
                # mux.append((t, msg))
                steering_angle.append((t, msg.drive.steering_angle))
                speed.append((t, msg.drive.speed))

        print('number of camera', len(camera), ' mux ', len(steering_angle))

        idxmux = 0

        trans = transfer_to_one_hot([-0.340000003576, 0.340000003576], 0.05)

        final_out = []
        for i in range(trans.oh_n):
            final_out.append([])

        print ('symcronizeing and data sampling')
        for idxImg in range(len(camera)):
            if idxImg % 100 == 0:
                print('------%d / %d------' % (idxImg, len(camera)))

            # if idxImg % 7 != 0:
            #     continue

            targetT = camera[idxImg][0]
            while idxmux < len(steering_angle) - 1 and steering_angle[idxmux][0] < targetT:
                idxmux += 1
            if idxmux > 0 and targetT - steering_angle[idxmux - 1][0] < steering_angle[idxmux][0] - targetT:
                idxmux -= 1

            if not speed[idxmux][1]:
                # print('Discard, speed is ', speed[idxmux][1])
                continue

            # str_ang = round(float(steering_angle[idxmux][1]),2)
            # spd = round(float(speed[idxmux][1]), 1)

            str_ang = trans.transter(float(steering_angle[idxmux][1]))

            # final_out[str_ang.index(1)].append([idxImg, np.array(str_ang)])
            final_out[str_ang.index(1)].append([idxImg, steering_angle[idxmux][1], speed[idxmux][1]])

        print(len(final_out), trans.count)

        # final_out = balance(trans.count, final_out)
        print('image load and processing')
        preprocess(final_out, PATH_TO_SAVE, bag_file)
