#!/usr/bin/env python


from scipy.spatial.transform import Rotation


def run_plan(pub_init_pose, pub_controls, plan):
    init = plan.pop(0)
    send_init_pose(pub_init_pose, init)

    for c in plan:
        send_command(pub_controls, c)

def Quaternion(y, p, r):
    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler('xyz', [y, p, r], degrees=True)

    # Convert to quaternions and print
    rot_quat = rot.as_quat()

    return rot_quat

def send_init_pose(pub_init_pose, init_pose):
    pose_data = init_pose.split(",")
    assert len(pose_data) == 3

    x, y, theta = float(pose_data[0]), float(pose_data[1]), float(pose_data[2])
    q = Quaternion(0, 0, theta)
    # point = Point(x=x, y=y)
    # pose = PoseWithCovariance(pose=Pose(position=point, orientation=q))
    # pub_init_pose.publish(PoseWithCovarianceStamped(pose=pose))

    print('initial positon is: %f, %f, orientation is: '%(x,y), q)

def send_command(pub_controls, c):
    cmd = c.split(",")
    assert len(cmd) == 2
    v, delta = float(cmd[0]), float(cmd[1])

    print('steering_angle is: %f, speed is: %f'%(delta,v))


if __name__ == "__main__":
    plan_file = '/home/li/Documents/prl-mushr/mushr_sim/plans/figure_8.txt'

    with open(plan_file) as f:
        plan = f.readlines()

    run_plan(None, None, plan)
