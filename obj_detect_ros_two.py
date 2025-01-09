# ------------------------------------------------------------------------------------------------------------
# =========================================== YOLOv5 ROS2 ====================================================
# ------------------------------------------------------------------------------------------------------------
import time
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.dataloaders import LoadStreams, LoadImages
from utils.augmentations import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_boxes, xyxy2xywh, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync


import numpy as np
from cv_bridge import CvBridge
from utils.image_publisher import *
from geometry_msgs.msg import Vector3Stamped, Point
from visualization_msgs.msg import Marker

# ------------------------------------------------------------------------------------------------------------
# Importing required ROS2 modules
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from ament_index_python.packages import get_package_share_directory

from boundingboxes.msg import BoundingBox, BoundingBoxes


class ImageStreamSubscriber(Node):

    def __init__(self):
        super().__init__('yolov5_ros2_node')
        
        # location of package
        package_share_directory = get_package_share_directory('yolov5_ros2')
        weight_loc = list()
        for direc in package_share_directory.split("/"):
            if direc != 'install' and direc != 'src' and direc != 'build':
                weight_loc.append(direc)
            else:
                break
        weight_loc.append("./valve_detect_yolo/weights/after_ward/special_a/")
        weight_loc = "/".join(weight_loc)
        #print(weight_loc)
        self.camera_info = None
        self.center = []

        # parameters
        self.declare_parameter('weights', 'best.pt')
        self.declare_parameter('subscribed_topic', '/camera/camera/color/image_raw')
        
        self.declare_parameter('published_topic', '/yolov5_ros2/image')

        self.declare_parameter('camera_info', '/camera/camera/color/camera_info')
        self.declare_parameter('direction_vector_topic', '/direction_vector')
        self.declare_parameter('img_size', 416)
        self.declare_parameter('device', '')
        self.declare_parameter('conf_thres', 0.35)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 1000)
        self.declare_parameter('classes', None)
        self.declare_parameter('hide_labels', False)
        self.declare_parameter('hide_conf', False)
        self.declare_parameter('augment', True)
        self.declare_parameter('agnostic_nms', True)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('half', False)
        self.declare_parameter('direction_marker_topic', '/direction_vector_marker') 
        
        self.weights =  str(weight_loc) + self.get_parameter('weights').value
        self.published_topic = self.get_parameter('published_topic').value
        self.subscribed_topic = self.get_parameter('subscribed_topic').value
        self.imgsz = self.get_parameter('img_size').value
        self.device = self.get_parameter('device').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.direction_vector_topic = self.get_parameter('direction_vector_topic').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.classes = self.get_parameter('classes').value
        self.hide_labels = self.get_parameter('hide_labels').value
        self.hide_conf = self.get_parameter('hide_conf').value
        self.augment = self.get_parameter('augment').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.half = self.get_parameter('half').value
        self.camera_info_subscriber = self.create_subscription(CameraInfo,self.get_parameter('camera_info').value,self.camera_info_callback,10 )
        self.direction_marker_topic = self.get_parameter('direction_marker_topic').value


        check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
        self.bridge = CvBridge()
        
        # loading model
        self.model_initialization()
        
        # initializing publish and subscribe nodes
        self.flag = ord('a')
        self.detection_img_pub = self.create_publisher(Image, self.published_topic, 10)
        self.bboxes_pub = self.create_publisher(BoundingBoxes,"yolov5_ros2/bounding_boxes", 10)
        self.direction_vector_pub = self.create_publisher(
            Vector3Stamped,
            self.direction_vector_topic,
            10
        )
        self.marker_pub = self.create_publisher(
            Marker,
            self.direction_marker_topic,
            10
        )
        
       
        self.subscription = self.create_subscription(Image, self.subscribed_topic, self.subscriber_callback, 10)

        self.subscription                                                           # prevent unused variable warning

    def camera_info_callback(self, msg):
        self.camera_info = msg


    def integration_with_lidar(self, center):
        if self.camera_info is None:
            print("------WARNING: NO CAMERA INFO FOUND------")
            return

        vector_msg = Vector3Stamped()
        vector_msg.header.stamp = self.get_clock().now().to_msg()
        vector_msg.header.frame_id = self.camera_info.header.frame_id
        K = np.array(self.camera_info.k).reshape(3, 3)
        D = np.array(self.camera_info.d)
        image_point = np.array([[center]], dtype=np.float32) 
        undistorted_points = cv2.undistortPoints(image_point, K, D)

        x_n = undistorted_points[0, 0, 0]
        y_n = undistorted_points[0, 0, 1]

        direction_vector = np.array([x_n, y_n, 1.0])

        direction_vector /= np.linalg.norm(direction_vector)
        vector_msg.vector.x = direction_vector[0]
        vector_msg.vector.y = direction_vector[1]
        vector_msg.vector.z = direction_vector[2]


        marker = Marker()
        marker.header.frame_id = self.camera_info.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "direction_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Define the start and end points of the arrow
        start_point = Point()
        start_point.x = 0.0  # Assuming camera origin
        start_point.y = 0.0
        start_point.z = 0.0

        # Define the end point based on the direction vector (scaled for visualization)
        scale = 0.4  # Adjust the scale as needed
        end_point = Point()
        end_point.x = direction_vector[0] * scale
        end_point.y = direction_vector[1] * scale
        end_point.z = direction_vector[2] * scale

        marker.points = [start_point, end_point]

        # Set the arrow's scale (shaft diameter and head diameter)
        marker.scale.x = 0.02  # Shaft diameter
        marker.scale.y = 0.04  # Head diameter
        marker.scale.z = 0.0  # Not used for ARROW

        # Set the arrow's color (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Alpha (opacity)

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()  # 0 means forever

        # Publish the marker
        self.marker_pub.publish(marker)
        # self.direction_vector_pub.publish(vector_msg)

        # print("Direction vector from camera center to object:", direction_vector)
        return vector_msg




    def subscriber_callback(self, msg):
        
        # storing input image msg header
        imgmsg_header = msg.header
        
        # converting image-ros-msg into 3-channel (bgr) image formate
        self.im0s = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Padded resize
        self.img = letterbox(self.im0s, self.imgsz, stride=self.stride)[0]
        
        # Convert
        self.img = self.img.transpose((2, 0, 1))[::-1]                              # HWC to CHW, BGR to RGB
        self.img = np.ascontiguousarray(self.img)
        
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.half() if self.half else self.img.float()               # uint8 to fp16/32
        self.img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)
        
        # Inference
        self.t1 = time_sync()
        self.pred = self.model(self.img, augment=self.augment)[0]

        # Apply NMS
        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        self.t2 = time_sync()

        # Apply Classifier
        if self.classify:
            self.pred = apply_classifier(self.pred, self.modelc, self.img, self.im0s)
        
        # BoundingBoxes msg
        bboxes = BoundingBoxes()
        
        # Process detections
        for i, det in enumerate(self.pred):  # detections per image
            s, im0 = '', self.im0s.copy()
            s += '%gx%g ' % self.img.shape[2:]  # print string

            if len(det):
            # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(self.img.shape[2:], det[:, :4], im0.shape).round()

                # Find the detection with the highest confidence
                max_conf_idx = torch.argmax(det[:, 4])
                highest_det = det[max_conf_idx]

                # Unpack the detection
                xyxy = highest_det[:4]
                conf = highest_det[4]
                cls = highest_det[5]

                # Convert tensor values to CPU and then to numpy for further processing
                xyxy = xyxy.cpu().numpy()
                conf = conf.cpu().item()
                cls = cls.cpu().item()

                # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.hide_labels else (
                    self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                plot_one_box(
                    xyxy,
                    im0,
                    label=label,
                    color=colors(c, True),
                    line_thickness=self.line_thickness - 1
                )

                # Single BoundingBox msg
                single_bbox = BoundingBox()
                single_bbox.xmin = int(xyxy[0])
                single_bbox.ymin = int(xyxy[1])
                single_bbox.xmax = int(xyxy[2])
                single_bbox.ymax = int(xyxy[3])
                single_bbox.probability = conf
                single_bbox.id = c
                single_bbox.class_id = self.names[c]

                # Calculate center coordinates
                x_center = (single_bbox.xmin + single_bbox.xmax) / 2
                y_center = (single_bbox.ymin + single_bbox.ymax) / 2

                center = np.array([x_center, y_center])
                center_l = [x_center, y_center]
                self.center.append(center_l)

                if len(self.center)>20:
                    x_sum = self.center[-10:]
                    i = 0
                    x_x = 0
                    y_y = 0
                    for i in range(10):
                        x_x += x_sum[i][0]
                        y_y += x_sum[i][1]
                    x_avg = x_x/10
                    y_avg = y_y/10
                    # center_p = np.array([x_avg, y_avg])
                    # distance = np.sqrt(((center[0] - center_p[0])**2) + ((center[1] - center_p[1])**2))
                   
                    center_p = np.array([x_avg, y_avg])
                    distance = np.sqrt(((center[0] - center_p[0])**2) + ((center[1] - center_p[1])**2))
                    # print(f"the distance is {distance}")
                    if distance<20:
                        vector_msg = self.integration_with_lidar(center)
                        # print("Direction vector from camera center to object:", direction_vector)
                        self.direction_vector_pub.publish(vector_msg)




                # print(f"center is {x_center} and {y_center}")

                # Optionally, add center coordinates to the BoundingBox message if fields are available
                # single_bbox.x_center = x_center
                # single_bbox.y_center = y_center

                bboxes.bounding_boxes.append(single_bbox)
            else:
                # Handle the case with no detections if necessary
                pass

        
        # Publishing bounding boxes and image with bounding boxes attached
        # same time-stamp for image and bounding box published, to match input image and output boundingboxes frames
        timestamp = (self.get_clock().now()).to_msg()
        
        processed_imgmsg = self.bridge.cv2_to_imgmsg(np.array(im0), encoding="bgr8")
        processed_imgmsg.header = imgmsg_header                                     # assigning header of input image msg
        processed_imgmsg.header.stamp = timestamp
        
        bboxes.header = imgmsg_header                                               # assigning header of input image msg
        bboxes.header.stamp = timestamp
        
        self.detection_img_pub.publish(processed_imgmsg)
        self.bboxes_pub.publish(bboxes)
        
        
    @torch.no_grad()
    def model_initialization(self):
        
        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.half and self.device.type != 'cpu'                         # half precision only supported on CUDA
        print("device:",self.device)
        
        # Load model
        self.model = attempt_load(self.weights, device=self.device)           # load FP32 model
        self.stride = int(self.model.stride.max())                                  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)                      # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        print("------------------Names of Classes------------------",self.names)
        
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)                    # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        
        return None
        


def main(args=None):
    rclpy.init(args=args)
    
    image_node = ImageStreamSubscriber()
    rclpy.spin(image_node)
    
    image_node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
