import os

import cv2
import random
from scipy import ndimage
import numpy as np
import math
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import shutil


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
        num = sum(count) / len(count)
        print('balance based on the average ration: ', num)

    new_count = [[]] * len(count)
    for ind, class_num in enumerate(count):
        print('---the %d class---' % (ind + 1))
        tmp = []

        if class_num >= num:
            resultList = random.sample(range(0, class_num), num)
            for i in resultList:
                tmp.append(final_out[ind][i])
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


def augmentation(img, if_rotate=True, if_shift=True, if_crop=True):
    # img = img.rotate(random.randint(0,15))
    if if_rotate:
        img = ndimage.rotate(img, random.randint(-15, 15), reshape=False)

    cord = [330, 24]
    if if_shift:
        cord = [random.randint(180, 370), random.randint(0, 48)]

    if if_crop:
        img = img[:, cord[0]:cord[0] + 360, cord[1]:cord[1] + 800]  # stanard = [330:690, 24:824]

    return img


def preprocess(data, path = None):
    final = []
    img_pth = []

    if path != None:
        for n in img_prefix:
            new_path = os.path.join(path, n)
            os.mkdir(new_path)
            img_pth.append(new_path)

    for ind, img_id in enumerate(data):
        if ind % 500 == 0:
            print('------%d / %d------' % (ind, len(data)))

        idx_list = img_id[0]

        ## image load
        for ind, n in enumerate(img_id):
            img = bridge.imgmsg_to_cv2(ms_list[n][idx_list[n]][1], "bgr8")
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # fisheye_image = np.concatenate([[fisheye1_image], [fisheye2_image]])
            # print('fisheye_image\'s shape', fisheye_image.shape)

            # fisheye_image = augmentation(fisheye_image, False, False, True)
            # print('fisheye_image\'s shape after augmentation', fisheye_image.shape)

            # cv2.imshow('image', fisheye1_image)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(img_pth[ind], img_prefix[ind] + str(idx_list[n]).zfill(5) + '.jpg'), img)

            # final.append([np.array(fisheye_image), d[1]])

    return final


def plot_chart(Y, X=None, title='Line Chart'):
    if X == None:
        plt.plot(Y)
    else:
        plt.plot(Y, X)
    plt.title(title)
    # plt.xlabel('Year')
    # plt.ylabel('Unemployment Rate')
    plt.show()


if __name__ == "__main__":
    bridge = CvBridge()  # ros img transfer to cv2

    PATH_TO_INPUT_BAG = '/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/rec1.bag'
    SAVE_PATH = '/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/processed'
    # FILE_PREFIX = 'mux_only'  # file name
    topics = [
        # '/car/cameraD435i/color/image_raw_throttle,',
        # '/car/cameraD435i/depth/image_rect_raw_throttle',
        # '/car/cameraT265/fisheye1/image_raw_throttle',
        # '/car/cameraT265/fisheye2/image_raw_throttle',
        '/car/camera/color/image_raw',
        '/car/mux/ackermann_cmd_mux/output',
        # '/car/scan',
    ]

    mux_ind = topics.index('/car/mux/ackermann_cmd_mux/output')
    img_ind = [0] #[0,1,2,3]
    img_prefix = []
    for i in img_ind:
        nm = topics[i].split('/')[-2]
        img_prefix.append(nm)

    ms_list = []
    for i in range(len(topics)):
        ms_list.append([])

    ## load data directly from rosbag
    print('read bag', PATH_TO_INPUT_BAG)
    bag = rosbag.Bag(PATH_TO_INPUT_BAG)

    print('read message', topics)
    for topic, msg, t in bag.read_messages(topics=topics):
        for i, top in enumerate(topics):
            if topic == top:
                if topic == '/car/mux/ackermann_cmd_mux/output':
                    ms_list[i].append((t, [msg.drive.steering_angle, msg.drive.speed]))
                    # steering_angle.append(msg.drive.steering_angle)
                    # speed.append(msg.drive.speed)
                else:
                    ms_list[i].append((t, msg))

    # plot_chart(steering_angle, title='steering angle')
    # plot_chart(speed, title='speed')

    ics_len = [len(i) for i in ms_list]
    for ind, l in enumerate(ics_len):
        print('number of %s, %d' % (topics[ind], l))

    idx_count = [0] * len(topics)

    trans = transfer_to_one_hot([-0.340000003576, 0.340000003576], 0.12)

    final_out = []
    for i in range(trans.oh_n):
        final_out.append([])
    # print(final_out)

    print('symcronizeing and data sampling')
    count = 0
    less_topic = min(ics_len)
    ind_les_top = ics_len.index(less_topic)
    for idx in range(less_topic):
        # if count >= 1:
        #     break

        if idx % 100 == 0:
            print('------%d / %d------' % (idx, less_topic))

        # if idx % 7 != 0:
        #     continue

        targetT = ms_list[ind_les_top][idx][0]
        while idx_count[mux_ind] < len(ms_list[mux_ind]) - 1 and ms_list[mux_ind][idx_count[mux_ind]][0] < targetT:
            idx_count[mux_ind] += 1
        if idx_count[mux_ind] > 0 and targetT - ms_list[mux_ind][idx_count[mux_ind] - 1][0] < \
                ms_list[mux_ind][idx_count[mux_ind]][0] - targetT:
            idx_count[mux_ind] -= 1

        if not ms_list[mux_ind][idx_count[mux_ind]][1][1]:
            # print('Discard, speed is ', speed[idxmux][1])
            continue

        for j in range(len(ms_list)):
            if j != mux_ind:
                if j == ind_les_top:
                    idx_count[ind_les_top] = idx
                else:
                    while idx_count[j] < len(ms_list[j]) - 1 and ms_list[j][idx_count[j]][0] < targetT:
                        idx_count[j] += 1
                    if idx_count[j] > 0 and targetT - ms_list[j][idx_count[j] - 1][0] < ms_list[j][idx_count[j]][0] - targetT:
                        idx_count[j] -= 1

        ## steering angle and speed load
        # str_ang = round(float(steering_angle[idxmux][1]),2)
        # spd = round(float(speed[idxmux][1]), 1)

        # count += 1
        str_ang = trans.transter(float(ms_list[mux_ind][idx_count[mux_ind]][1][0]))

        final_out[str_ang.index(1)].append([idx_count, np.array(str_ang)])
        # final_out.append([idx_count, np.array(str_ang)])

    print(len(final_out), trans.count)
    print('check each class is same as count:')
    for i in range(len(final_out)):
        print('---class %d have %d data: '%(i+1, len(final_out[i])))

    final_out = balance(trans.count, final_out, 'max')
    # final_out = np.concatenate(final_out, axis=0)

    print('image load and processing')
    for i in range(trans.oh_n):
        new_dir = os.path.join(SAVE_PATH, str(i))
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.mkdir(new_dir)
        preprocess(final_out, new_dir)

    # final_out = preprocess(final_out)

    ## concateneta data
    final_out = np.array(final_out)
    # np.random.shuffle(final_out)
    print(final_out.shape)

    try:
        np.save('data/0810175158/' + FILE_PREFIX + '.npy', final_out)
        print('save to:', 'data/0810175158/' + FILE_PREFIX + '.npy')
    except:
        print('save faild')
