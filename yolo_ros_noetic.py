#!/usr/bin/env python3

import rospy
import torch
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError

def main():
    rospy.init_node('yolo_detector', anonymous=True)

    # Load the YOLOv5 model
    model = torch.hub.load('./yolov5', 'custom', path='./weights/for_drone_a/best.pt', source='local')  # Adjust path if necessary
    model.eval()

    bridge = CvBridge()

    # Publishers
    image_pub = rospy.Publisher('/yolov5/detections', Image, queue_size=1)
    bbox_centers_pub = rospy.Publisher('/yolov5/bbox_centers', Int32MultiArray, queue_size=1)

    def image_callback(msg):
        try:
            # Convert ROS Image message to OpenCV image
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Convert BGR to RGB for the model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img_rgb)

        # Parse results
        detections = results.pandas().xyxy[0]  # Pandas DataFrame

        centers = []
        # Iterate over detections
        for index, row in detections.iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            conf = row['confidence']
            cls = int(row['class'])
            name = row['name']

            # Draw rectangle and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Calculate and collect the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            centers.extend([center_x, center_y])
            rospy.loginfo(f"Center of bbox: ({center_x}, {center_y})")

        try:
            # Convert OpenCV image back to ROS Image message
            img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
            image_pub.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # Publish centers of bounding boxes if any
        if centers:
            centers_msg = Int32MultiArray(data=centers)
            bbox_centers_pub.publish(centers_msg)

    # Subscribe to the image topic (change '/camera/image_raw' to your image topic)
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    main()

