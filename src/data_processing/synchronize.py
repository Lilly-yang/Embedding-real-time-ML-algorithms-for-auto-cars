from functions import *
from copy import deepcopy


if __name__ == "__main__":
    MSG_PATH = '/home/li/Documents/sensor_data/data/MIA/s1_p4_1/topics'
    SAVE_PATH = '/home/li/Documents/sensor_data/data/MIA/s1_p4_1/sync'
    mk_dir(SAVE_PATH)

    topics = [
        # '/car/cameraD435i/color/image_raw_throttle,',
        '/car/cameraD435i/depth/image_rect_raw_throttle',
        '/car/cameraT265/fisheye1/image_raw_throttle',
        '/car/cameraT265/fisheye2/image_raw_throttle',
        '/car/mux/ackermann_cmd_mux/output',
        # '/car/scan',
    ]

    mux_ind = topics.index('/car/mux/ackermann_cmd_mux/output')
    dir_name = []
    for t in topics:
        name = t.split('/')[-2]
        mk_dir(os.path.join(SAVE_PATH, name))
        dir_name.append(name)

    ms_list = []
    for i in range(len(topics)):
        ms_list.append([])

    print('read message', topics)
    for ind, msg in enumerate(dir_name):
        if 'ackermann_cmd_mux' in msg:
            with open(os.path.join(MSG_PATH, msg, 'steering_angle_and_speed')) as f:
                for line in f.readlines():
                    t, angle, speed = line.split(',')
                    ms_list[mux_ind].append((int(t), (round(float(angle), 2), round(float(speed), 2))))
        else:
            img_path_list = os.listdir(os.path.join(MSG_PATH, msg))
            img_path_list.sort()
            for f in img_path_list:
                ms_list[ind].append(int(f.split('.')[0]))

    # plot_chart(steering_angle, title='steering angle')
    # plot_chart(speed, title='speed')

    ics_len = [len(i) for i in ms_list]
    for ind, l in enumerate(ics_len):
        print('number of %s, %d' % (topics[ind], l))
    less_topic = min(ics_len)
    ind_les_top = ics_len.index(less_topic)

    idx_count = [0] * len(topics)

    final_out = []

    print('symcronizeing and data sampling: %d messages' % less_topic)
    for idx in range(less_topic):
        if idx % 500 == 0:
            print('------%d / %d------' % (idx, less_topic))

        targetT = ms_list[ind_les_top][idx]
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
                    idx_count[j] = idx
                else:
                    while idx_count[j] < len(ms_list[j]) - 1 and ms_list[j][idx_count[j]] < targetT:
                        idx_count[j] += 1
                    if idx_count[j] > 0 and targetT - ms_list[j][idx_count[j] - 1] < ms_list[j][idx_count[j]] - targetT:
                        idx_count[j] -= 1

        hard_copy = deepcopy(idx_count)
        final_out.append(hard_copy)

    print('image load and processing from %d messages' % len(final_out))
    for idx, msg_id in enumerate(final_out):
        if idx % 500 == 0:
            print('------%d / %d------' % (idx, len(final_out)))

        steering_angle = ms_list[3][msg_id[3]][1][0]
        speed = ms_list[3][msg_id[3]][1][1]
        for i in range(3):
            t = ms_list[i][msg_id[i]]
            shutil.copyfile(os.path.join(MSG_PATH, dir_name[i], str(t) + '.jpg'),
                            os.path.join(SAVE_PATH, dir_name[i],
                                         str(t) + '_' + str(steering_angle) + '_' + str(speed) + '.jpg'))
