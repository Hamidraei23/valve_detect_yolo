
# **YOLOv5 Valve Detector**

This project offers a containerized environment to efficiently deploy deep learning models using PyTorch and YOLOv5. It is designed to ensure compatibility, reproducibility, and ease of deployment across diverse systems, with a specific focus on valve detection tasks.

---

## **Overview**

This repository leverages a Dockerized setup to simplify the development and deployment of YOLOv5 models. By abstracting the complexities of dependency management and environment configuration, this approach enables users to focus on training and utilizing the model without worrying about system-specific issues.

Key highlights:
- A **Dockerfile** pre-configured with all necessary dependencies for PyTorch and YOLOv5.
- **ONNX export** for efficient deployment and compatibility across CPU-only systems.
- A sample ROS implementation for real-time valve detection.

---

## **Features**

- **Preconfigured Docker Environment**: A streamlined Dockerfile that sets up a consistent development environment with PyTorch and YOLOv5.
- **ONNX Export**: The model is exported to ONNX for lightweight deployment on CPU-only systems.
- **ROS Integration**: Example Python scripts demonstrate how to use the YOLOv5 model in ROS-based applications.
- **GPU and CPU Compatibility**: Support for GPU acceleration (via PyTorch) and CPU-only execution for flexible deployment.

---

## **Getting Started**

### **Cloning the Repository**
1. Clone the YOLOv5 repository:
   ```bash
   git clone git@github.com:ultralytics/yolov5.git
   cd yolov5
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Using the Docker Environment**

### **Build the Docker Image**
To build the Docker image, use the following command:
```bash
docker build -t yolov5-valve-detector .
```

### **Run the Docker Container**
To run the container with GPU support, use:
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace yolov5-valve-detector
```

---

## **Model Usage**

### **Inference with YOLOv5**
Run the following command for inference using the trained YOLOv5 model:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source /path/to/images
```

### **ONNX Model Execution**
The trained YOLOv5 model has been exported to ONNX for lightweight deployment. A sample script is included to demonstrate how to run the ONNX model on a CPU:
```bash
python onnx_inference.py --model-path models/best.onnx --input /path/to/images
```

---

## **ROS Integration**

For real-time valve detection using ROS, the provided Python script performs the following:
- Subscribes to a ROS topic as input.
- Processes the input through the YOLOv5 model.
- Publishes annotated outputs or messages showing the detected center coordinates.

To use the ROS node, refer to the example script:
```bash
rosrun valve_detection ros_node_sample.py
```

You can adapt this script to your specific ROS setup as needed.

---

## **Demo**

### **YouTube Video Demo**
Click the image below to view the valve detection demo on YouTube:
[![Demo Video](https://img.youtube.com/vi/CxQ9xXzTJyQ/0.jpg)](https://www.youtube.com/watch?v=CxQ9xXzTJyQ)

### **GIF Demo**
A quick GIF demonstration is also available:
![Demo GIF](demo.gif)

---

## **Development Details**

### **Training and Export**
- The model was trained using PyTorch and YOLOv5.
- It has been exported to ONNX for CPU-based deployments and can also be used on GPU-enabled systems.
- The processed output provides annotations or a message indicating the center of the detected valve, making it adaptable for ROS topics or other application-specific needs.

### **GPU-Enabled Execution**
For GPU-based applications, refer to `yolo_gpu.py` for an example of implementing the model in GPU-enabled environments.

---

## **Contributing**
We welcome contributions to improve this project! Please fork the repository and submit a pull request with your changes. Make sure to adhere to the coding and documentation standards.

---

## **License**
This project is licensed under the [MIT License](LICENSE). Please refer to the LICENSE file for more details.

---

If you have any questions or need further assistance, feel free to raise an issue in the repository.
