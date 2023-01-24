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


def preprocess(data):
    final = []

    for ind, d in enumerate(data):
        # if ind % 1000 == 0:
        #     print('------%d / %d------'%(ind, len(data)))
        print('------%d / %d------'%(ind, len(data)))

        new_dir = os.path.join('/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/processed/', str(ind))
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.mkdir(new_dir)

        for msg in d:
            idxImg = int(msg[0])

        ## image load
        # fisheye1_image = bridge.imgmsg_to_cv2(fisheye1[idxImg][1], "bgr8")
        # fisheye2_image = bridge.imgmsg_to_cv2(fisheye2[idxImg][1], "bgr8")
            camera_image = bridge.imgmsg_to_cv2(camera[idxImg][1], "bgr8")

        # # prepocessing
        # fisheye1_image = cv2.cvtColor(fisheye1_image, cv2.COLOR_BGR2GRAY)
        # fisheye2_image = cv2.cvtColor(fisheye2_image, cv2.COLOR_BGR2GRAY)

        # fisheye1_image = augmentation(fisheye1_image)
        # fisheye2_image = augmentation(fisheye2_image)
            camera_image = augmentation(camera_image)

        # cv2.imshow('image', fisheye1_image)
        # cv2.waitKey(0)

        # cv2.imwrite('/home/li/Documents/sensor_data/python_data/' + FILE_PREFIX + str(idxImg).zfill(5) + '_fisheye1_' + str(
        #     servo[idxsv][1])[5:10] + '.jpg',
        #             fisheye1_image)
        # cv2.imwrite('/home/li/Documents/sensor_data/python_data/' + FILE_PREFIX + str(idxImg).zfill(5) + '_fisheye2_' + str(
        #     servo[idxsv][1])[5:10] + '.jpg',
        #             fisheye2_image)
            cv2.imwrite(new_dir+'/'+str(idxImg).zfill(5)+'camera'+'.jpg', camera_image)
            print('save to:', new_dir+'/'+str(idxImg).zfill(5)+'camera'+'.jpg')

        ## conbine data
        # output_1.append([np.array(fisheye1_image), np.array((float(steering_angle[idxmux][1]), float(speed[idxmux][1])))])
        # output_2.append([np.array(fisheye2_image), np.array((float(steering_angle[idxmux][1]), float(speed[idxmux][1])))])

        # final.append([np.array(fisheye1_image), d[1]])
        # final.append([np.array(fisheye2_image), d[1]])
        # final.append([np.array(camera_image), d[1]])
        # final.append([camera_image, d[1]])

    return final


if __name__ == "__main__":
    PATH_TO_INPUT_BAG = '/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/rec1.bag'
    FILE_PREFIX = 'fisheyes_steering_croped_one_hot_balanced_test'  # file name
    topics = [#'/car/camera/fisheye1/image_raw',
            #   '/car/camera/fisheye2/image_raw',
              '/car/mux/ackermann_cmd_mux/output',
              # '/car/vesc/commands/servo/position',
              '/car/camera/color/image_raw',
              ]

    # load data directly from rosbag
    print('read bag', PATH_TO_INPUT_BAG)
    bag = rosbag.Bag(PATH_TO_INPUT_BAG)

    # fisheye1 = []
    # fisheye2 = []
    camera = []
    # motor = []
    # servo = []
    # mux = []
    steering_angle = []
    speed = []

    print('read message', topics)
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == '/car/camera/color/image_raw':  # 30/s
            camera.append((t, msg))
        # elif topic == '/car/camera/fisheye2/image_raw':  # 30/s
        #     fisheye2.append((t, msg))
        # elif topic == '/car/vesc/commands/motor/speed':  # 116/s
        #     motor.append((t, msg))
        # elif topic == '/car/vesc/commands/servo/position':  # 75/s
        #     servo.append((t, msg))
        elif topic == '/car/mux/ackermann_cmd_mux/output':  # 63/s
            # mux.append((t, msg))
            steering_angle.append((t, msg.drive.steering_angle))
            speed.append((t, msg.drive.speed))

    print('number of camera', len(camera), ' mux ', len(steering_angle))

    # idxmt = 0
    # idxsv = 0
    idxmux = 0
    # idxstr_ang = 0
    # idxspd = 0
    output_1 = []
    output_2 = []

    trans = transfer_to_one_hot([-0.340000003576, 0.340000003576], 0.06)

    final_out = []
    for i in range(trans.oh_n):
        final_out.append([])
    # print(final_out)

    print ('symcronizeing and data sampling')
    count = 0
    for idxImg in range(len(camera)):
        # if count >= 1:
        # break

        if idxImg % 100 == 0:
            print('------%d / %d------' % (idxImg, len(camera)))

        # if idxImg % 7 != 0:
        #     continue

        targetT = camera[idxImg][0]
        # while idxmt < len(motor) - 1 and motor[idxmt][0] < targetT: idxmt += 1
        # while idxsv < len(servo) - 1 and servo[idxsv][0] < targetT: idxsv += 1
        while idxmux < len(steering_angle) - 1 and steering_angle[idxmux][0] < targetT: idxmux += 1
        # while idxspd < len(speed) - 1 and speed[idxspd][0] < targetT: idxspd += 1

        # if idxmt > 0 and targetT - motor[idxmt - 1][0] < motor[idxmt][0] - targetT:
        #     idxmt -= 1
        # if idxsv > 0 and targetT - servo[idxsv - 1][0] < servo[idxsv][0] - targetT:
        #     idxsv -= 1
        if idxmux > 0 and targetT - steering_angle[idxmux - 1][0] < steering_angle[idxmux][0] - targetT:
            idxmux -= 1

        # print(speed[idxmux][1])
        # if not speed[idxmux][1]:
        #     # print('Discard, speed is ', speed[idxmux][1])
        #     continue

        # print(idxImg, idxmt, idxsv)
        # print(targetT, motor[idxmt][0], servo[idxsv][0])
        # print(targetT, motor[idxmt][1], servo[idxsv][1])
        # print(round(float(steering_angle[idxmux][1]),2), round(float(speed[idxmux][1])))

        ## steering angle and speed load
        # str_ang = round(float(steering_angle[idxmux][1]),2)
        # spd = round(float(speed[idxmux][1]), 1)

        # count += 1
        str_ang = trans.transter(float(steering_angle[idxmux][1]))

        # final_out[str_ang.index(1)].append([idxImg, np.array(str_ang)])
        final_out[str_ang.index(1)].append([idxImg, steering_angle[idxmux][1]])

    print(len(final_out), trans.count)
    # print('check each class is same as count:')
    # for i in range(len(final_out)):
    #     print('---class %d have %d data: '%(i+1, len(final_out[i])))

    final_out = balance(trans.count, final_out)
    # final_out = np.concatenate(final_out, axis=0)
    print('image load and processing')
    final_out = preprocess(final_out)

    ## concateneta data
    # final_out = np.array(final_out)
    # np.random.shuffle(final_out)
    # print(final_out.shape)

    # try:
    #     np.save('/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/processed/' + FILE_PREFIX + '.npy', final_out)
    #     print('save to:', '/root/catkin_ws/src/Embedding-real-time-ML-algorithms-for-auto-cars/data/IMREDD/processed' + FILE_PREFIX + '.npy')
    # except:
    #     print('save faild')
