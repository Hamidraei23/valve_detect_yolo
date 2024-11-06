import onnxruntime as ort
import numpy as np
import cv2

# Load the ONNX model
ort_session = ort.InferenceSession("best.onnx")

# Load an image and preprocess it for YOLOv5
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))  # resize to YOLOv5 input size
    img_normalized = img_resized / 255.0  # normalize
    img_normalized = img_normalized.transpose(2, 0, 1)  # change shape to (3, 640, 640)
    img_normalized = img_normalized[np.newaxis, :, :, :].astype(np.float32)  # add batch dimension
    return img, img_normalized 

image_path = "frame_000421.jpg"
original_img, input_tensor = preprocess_image(image_path)

# Run inference
outputs = ort_session.run(None, {"images": input_tensor})

# Process outputs (example)
detections = outputs[0]  # YOLOv5 model outputs bounding boxes
print(f"detection {(detections)}")
for detection in detections:
    if len(detection) >= 6:
        x_center, y_center, width, height, class_id, confidence = detection[:6]  # Extract first 6 values
        print(detection[2500])
        confidence = confidence[-1]

        if confidence > 0.5:  # Filter out low-confidence detections
            # Convert YOLO format to bounding box coordinates
            x1 = int((x_center - width / 2) * original_img.shape[1])
            y1 = int((y_center - height / 2) * original_img.shape[0])
            x2 = int((x_center + width / 2) * original_img.shape[1])
            y2 = int((y_center + height / 2) * original_img.shape[0])

            # Draw rectangle and label on the image
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class: {int(class_id)}, Conf: {confidence:.2f}"
            cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate and print the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            print(f"Center of bbox: ({center_x}, {center_y})")



# Show the annotated image
cv2.imshow("YOLOv5 Detection", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()