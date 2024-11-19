import os
import glob
import torch
import cv2
import numpy as np

def main():
    # Load the YOLOv5 model
    model = torch.hub.load('./yolov5', 'custom', path='./weights/for_drone_a/best.pt', source='local')  # Adjust path if necessary

    # Set model to evaluation mode
    model.eval()

    # Specify the folder containing images
    # Replace 'best.pt' with the path to your trained model weights

    # Specify the folder containing images
    image_folder = '/home/hami/workspaces/ros/noetic/test_ws/valve_detect_yolo/frame_test'

    # Get list of image files (modify extensions if needed)
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg')) + \
                  glob.glob(os.path.join(image_folder, '*.png')) + \
                  glob.glob(os.path.join(image_folder, '*.jpeg'))

    # Loop over each image in the folder
    for image_path in image_paths:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_path}")
            continue

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model(img_rgb)

        # Parse results
        detections = results.pandas().xyxy[0]  # Pandas DataFrame

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

            # Calculate and print the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            print(f"Image: {os.path.basename(image_path)} - Center of bbox: ({center_x}, {center_y})")

        # Show the annotated image
        cv2.imshow("YOLOv5 Detection", img)

        # Wait for a key press and check for 'q' key to exit
        if cv2.waitKey(120) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
