import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from ultralytics import YOLO  # Import the YOLO class from ultralytics
import numpy as np
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Load YOLO model using ultralytics
        self.model = YOLO('/home/hami/workspaces/ros_two/yolo_ws/valve_detect_yolo/weights/best.pt')  # Update with your model path

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # Update with your image topic
            self.image_callback,
            10
        )

        # Create publisher for object center
        self.center_pub = self.create_publisher(Point, '/object_center', 10)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Run inference using the ultralytics YOLO model
        results = self.model(img_bgr)

        # Process results
        im0 = img_bgr.copy()
        annotator = Annotator(im0)
        highest_confidence = 0
        center_point = None

        for result in results:
            boxes = result.boxes
            if boxes:
                for box in boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    xyxy = box.xyxy.squeeze().cpu().numpy()
                    label = f"{self.model.names[cls]} {conf:.2f}"

                    # Draw bounding box
                    annotator.box_label(xyxy, label, color=colors(cls, True))

                    if conf > highest_confidence:
                        highest_confidence = conf
                        # Calculate center point
                        x1, y1, x2, y2 = xyxy
                        x_center = (x1 + x2) / 2.0
                        y_center = (y1 + y2) / 2.0
                        center_point = Point()
                        center_point.x = float(x_center)
                        center_point.y = float(y_center)
                        center_point.z = 0.0  # Assuming 2D image, z=0

        # Publish the center point of the object with highest confidence
        if center_point is not None:
            self.center_pub.publish(center_point)

        # Display image with detections
        im0 = annotator.result()
        cv2.imshow("YOLO Detection", im0)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
