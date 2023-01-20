"""
read rosbag need python 2.0
"""
import sys
import cv2
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import csv

sys.path.insert(1, '/home/li/Documents/sensor_data/src')
from functions import *


def read_Image_msg(msg, img_type='bgr8'):
    try:
        img = bridge.imgmsg_to_cv2(msg, img_type)
    except:
        print('wrong depth img', topic)
        exit()

    return img


if __name__ == "__main__":
    bridge = CvBridge()  # ros img transfer to cv2

    PATH_TO_INPUT_BAG = '/home/li/Documents/sensor_data/data/MIA/s1_p4_1.bag'
    SAVE_PATH = '/home/li/Documents/sensor_data/data/MIA/s1_p4_1/topics'
    mk_dir(SAVE_PATH)

    topics = [
        '/car/cameraD435i/color/image_raw_throttle,',
        '/car/cameraD435i/depth/image_rect_raw_throttle',
        '/car/cameraT265/fisheye1/image_raw_throttle',
        '/car/cameraT265/fisheye2/image_raw_throttle',
        '/car/mux/ackermann_cmd_mux/output',
        # '/car/scan',
    ]

    ms_path = []
    for t in topics:
        path = os.path.join(SAVE_PATH, t.split('/')[-2])
        ms_path.append(path)
        mk_dir(path)

    ## load data directly from rosbag
    print('read bag', PATH_TO_INPUT_BAG)
    bag = rosbag.Bag(PATH_TO_INPUT_BAG)

    count = 0
    count_all = [0] * len(topics)
    print('read message', topics)
    for topic, msg, t in bag.read_messages(topics=topics):
        count += 1
        if count % 100 == 0:
            print('finished %d messages' % count)

        for ind, top in enumerate(topics):
            if topic == top:
                count_all[ind] += 1
                if topic == '/car/mux/ackermann_cmd_mux/output':
                    with open(os.path.join(SAVE_PATH, 'ackermann_cmd_mux', 'steering_angle_and_speed'), 'a') as f:
                        write = csv.writer(f)
                        write.writerow([t, msg.drive.steering_angle, msg.drive.speed])
                else:
                    if topic == '/car/cameraD435i/depth/image_rect_raw_throttle':
                        img = read_Image_msg(msg, "32FC1")
                    else:
                        img = read_Image_msg(msg, "bgr8")

                    cv2.imwrite(os.path.join(ms_path[ind], str(t) + '.jpg'), img)
