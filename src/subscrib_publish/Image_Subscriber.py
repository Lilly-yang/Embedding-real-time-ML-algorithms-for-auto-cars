#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append("../src/training_testing")
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from pytorch_high_level import *
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped

# from geometry_msgs.msg import (
#     Point,
#     Pose,
#     PoseWithCovariance,
#     PoseWithCovarianceStamped,
#     Quaternion,
# )
# from tf.transformations import quaternion_from_euler

model_name = 'training_testing/models/steering.h5'
checkpoint = torch.load(model_name)
model = checkpoint['net']
model = model.to(device)

classes = [-0.34, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20,
           0.25, -0.30, -0.34]


class image_converter:
    def __init__(self):
        # self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/car/camera/fisheye1/image_raw_throttle", Image, self.callback)
        self.control_topic = rospy.get_param("~control_topic", "/car/mux/ackermann_cmd_mux/input/navigation")
        self.pub_controls = rospy.Publisher(self.control_topic, AckermannDriveStamped, queue_size=1)    # only need the latest msg

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        ## print imgs
        # (rows, cols, channels) = cv_image.shape
        # if cols > 60 and rows > 60 :
        #   cv2.circle(cv_image, (50,50), 10, 255)

        # cv2.imshow("Image window", cv_image)
        # cv2.waitKey(30)

        ## predict steering
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img = img[330:690, 24:824]
        img = torch.Tensor(img).view(-1, 1, 360, 800).to(device)
        ot = model(img)[0]
        ot = ot.cpu().data.numpy()
        steering_angle = classes[int(np.where(ot == np.amax(ot))[0])]
        # print('prediction: ', output)

        # try:
        #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except CvBridgeError as e:
        #   print(e)

        # dur = rospy.Duration(1.0)
        rate = rospy.Rate(10)
        start = rospy.Time.now()

        try:
            drive = AckermannDrive(steering_angle=steering_angle, speed=1.0)
            # self.pub_controls.publish(AckermannDriveStamped(drive=drive))
            # while rospy.Time.now() - start < dur:
            while not rospy.is_shutdown():
                self.pub_controls.publish(AckermannDriveStamped(drive=drive))
                rate.sleep()
        except CvBridgeError as e:
            print(e)


def main(args):
    rospy.init_node('image_converter', anonymous=True)  # anonymous: no same name

    image_converter()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
