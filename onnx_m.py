import onnxruntime as ort
import numpy as np
import cv2
import os
import glob

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))  # YOLOv5 input size
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    img_transposed = img_normalized.transpose(2, 0, 1)  # Channels first
    input_tensor = img_transposed[np.newaxis, :, :, :].astype(np.float32)  # Add batch dimension
    return img, input_tensor

def postprocess(outputs, conf_threshold=0.5, iou_threshold=0.5):
    # Output is a list with one element
    predictions = outputs[0]  # (batch_size, num_predictions, no)

    # Remove batch dimension
    predictions = np.squeeze(predictions, axis=0)  # (num_predictions, no)

    # Extract boxes, scores, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over each prediction
    for pred in predictions:
        # Extract the confidence (objectness score) and class probabilities
        objectness = pred[4]
        class_scores = pred[5:]

        # Multiply objectness score with class probabilities to get the final confidence
        scores = objectness * class_scores

        # Get the class with the highest confidence
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            # Convert from center coordinates to corner coordinates
            x_center, y_center, width, height = pred[0:4]
            x1 = (x_center - width / 2)
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)

            # Scale boxes to the input image size
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    # Prepare the final detections
    final_boxes = []
    final_confidences = []
    final_class_ids = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids[i])

    return final_boxes, final_confidences, final_class_ids

def main():
    # Load the ONNX model
    ort_session = ort.InferenceSession("best.onnx")

    # Specify the folder containing images
    image_folder = "/home/hami/workspaces/ros_two/summit_arm_drone_ws/FFF/FFF_b/frames"  # Replace with your folder path

    # Get list of image files (modify extensions if needed)
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.png")) + \
                  glob.glob(os.path.join(image_folder, "*.jpeg"))

    # Loop over each image in the folder
    for image_path in image_paths:
        # Preprocess the image
        original_img, input_tensor = preprocess_image(image_path)

        # Run inference
        outputs = ort_session.run(None, {"images": input_tensor})

        # Post-process the outputs
        boxes, confidences, class_ids = postprocess(outputs, conf_threshold=0.5, iou_threshold=0.5)

        # Draw boxes on the image
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box

            # Scale coordinates to original image size
            x_scale = original_img.shape[1] / 640
            y_scale = original_img.shape[0] / 640
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            # Draw rectangle and label on the image
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.putText(original_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            # Calculate and print the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            print(f"Image: {os.path.basename(image_path)} - Center of bbox: ({center_x}, {center_y})")

        # Show the annotated image
        cv2.imshow("YOLOv5 Detection", original_img)

        # Display the image for 1 millisecond and check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

