import cv2
import random
from scipy import ndimage
import math
from functions import *


class transfer_to_one_hot():
    def __init__(self, rng, step):
        self.oh_n = math.ceil((rng[1] - rng[0]) / step)
        if self.oh_n == (rng[1] - rng[0]) / step:
            self.oh_n += 1
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


def augmentation(img, crop=None, shift=(0,0), rotate=None):
    if rotate != None:
        img = ndimage.rotate(img, random.randint(rotate[0], rotate[1]), reshape=False)
    if crop != None:
        x,y = img.shape
        corp_cord = [int((x-crop[0])/2) + random.randint(-shift[0], shift[0]), int((y-crop[1])/2) + random.randint(-shift[1], shift[1])]
        img = img[corp_cord[0]:corp_cord[0] + crop[0], corp_cord[1]:corp_cord[1] + crop[1]]  # stanard = [330:690, 24:824]

    return img


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
    INPUT_PATH = '/home/li/Documents/sensor_data/data/MIA/s1_p4_1/sync'
    SAVE_PATH = '/home/li/Documents/sensor_data/data/MIA/s1_p4_1/train'
    mk_dir(SAVE_PATH)

    topics = [
        # '/car/cameraD435i/color/image_raw_throttle,',
        '/car/cameraD435i/depth/image_rect_raw_throttle',
        '/car/cameraT265/fisheye1/image_raw_throttle',
        '/car/cameraT265/fisheye2/image_raw_throttle',
        # '/car/mux/ackermann_cmd_mux/output',
        # '/car/scan',
    ]

    dir_name = []
    for t in topics:
        name = t.split('/')[-2]
        dir_name.append(name)

    trans = transfer_to_one_hot([-0.340000003576, 0.340000003576], 0.17)

    final_out = []
    for i in range(trans.oh_n):
        final_out.append([])
        mk_dir(os.path.join(SAVE_PATH, str(i)))

    for name in dir_name:
        for file in os.listdir(os.path.join(INPUT_PATH, name)):
            steering_angle = float(file.split('_')[1])
            lable = trans.transter(steering_angle)

            final_out[lable.index(1)].append(os.path.join(INPUT_PATH, name, file))

    print('check each class is same as count:')
    for i in range(len(final_out)):
        print('---class %d have %d data: '%(i+1, len(final_out[i])))

    final_out = balance(trans.count, final_out, 'max')

    print('image load and processing')
    for i in range(trans.oh_n):
        new_dir = os.path.join(SAVE_PATH, str(i))
        mk_dir(new_dir)

    for ind, files in enumerate(final_out):
        print('processing %d class'%ind)
        for id, path in enumerate(files):
            if id % 100 == 0:
                print('---%d / %d---'%(id, len(files)))
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = augmentation(img, crop=[360, 800], shift=(60,24))
            cv2.imwrite(os.path.join(SAVE_PATH, str(ind), str(id) + '_' + os.path.split(path)[-1]), img)
