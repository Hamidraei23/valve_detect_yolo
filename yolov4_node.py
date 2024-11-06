#!/usr/bin/env python

import rospy
import onnxruntime as ort
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

def preprocess_image(cv_image):
    img_resized = cv2.resize(cv_image, (640, 640))  # Resize to YOLOv5 input size
    img_normalized = img_resized / 255.0            # Normalize pixel values
    img_transposed = img_normalized.transpose(2, 0, 1)  # Channels first
    input_tensor = img_transposed[np.newaxis, :, :, :].astype(np.float32)  # Add batch dimension
    return input_tensor

def postprocess(outputs, conf_threshold=0.5, iou_threshold=0.5):
    predictions = outputs[0]  # Output from the model
    predictions = np.squeeze(predictions, axis=0)  # Remove batch dimension

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        objectness = pred[4]
        class_scores = pred[5:]
        scores = objectness * class_scores
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            x_center, y_center, width, height = pred[:4]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    final_boxes = []
    final_confidences = []
    final_class_ids = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids[i])

    return final_boxes, final_confidences, final_class_ids

class YOLOv5Node:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolov5_node', anonymous=True)

        # Load the ONNX model
        self.ort_session = ort.InferenceSession("best.onnx")

        # Create a CvBridge object
        self.bridge = CvBridge()

        # Subscribe to the /image_raw/color topic
        rospy.Subscriber("/image_raw/color", Image, self.image_callback)

        # Publishers
        self.image_pub = rospy.Publisher("/detect", Image, queue_size=1)
        self.center_pub = rospy.Publisher("/bbox_centers", Int32MultiArray, queue_size=1)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Preprocess the image
        input_tensor = preprocess_image(cv_image)

        # Run inference
        outputs = self.ort_session.run(None, {"images": input_tensor})

        # Post-process the outputs
        boxes, confidences, class_ids = postprocess(outputs, conf_threshold=0.5, iou_threshold=0.5)

        # Draw boxes on the image and collect centers
        centers = []

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box

            # Scale coordinates to original image size
            x_scale = cv_image.shape[1] / 640
            y_scale = cv_image.shape[0] / 640
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            # Draw rectangle and label on the image
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Calculate the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            centers.extend([center_x, center_y])

            rospy.loginfo(f"Detected object at center: ({center_x}, {center_y})")

        # Convert annotated image back to ROS Image message
        try:
            detect_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Publish the annotated image
        self.image_pub.publish(detect_msg)

        # Publish the centers of bounding boxes
        if centers:
            center_msg = Int32MultiArray(data=centers)
            self.center_pub.publish(center_msg)

def main():
    yolo_node = YOLOv5Node()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down YOLOv5 node")

if __name__ == "__main__":
    main()
