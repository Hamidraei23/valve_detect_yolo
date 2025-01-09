


## yolov5 Valve detector

This project provides a containerized environment to seamlessly run PyTorch and YOLO for deep learning tasks. Our setup ensures compatibility and ease of deployment across various systems.

# Overview
We start with a Dockerfile that sets up an environment with all necessary dependencies for PyTorch and YOLO, ensuring stability and reproducibility. This approach allows users to bypass the complexities of configuring dependencies and focus on developing and training models.

# Features
- Ready-to-use Dockerfile: Streamlined for PyTorch and YOLO usage.
- Dependency Management: Automatically installs all required libraries and dependencies.
- System Compatibility: Ensures a consistent environment across different machines.

# Run within the Docker
## clone yolov5

### git clone git@github.com:ultralytics/yolov5.git

### cd yolov5

### pip install -r requirements.txt 

## Sample command
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source /path/to/images


<<<<<<< HEAD
## The model has been transferred to ONNX, can be run using only CPU, a sample on how to run it is uploaded as well, 

## Use the sample to run rosnode for valve detection

=======
## Usage 	
the code model is exported to ONNX and a python file with sample of having ros topic as input is uploaded take a look and implement for your use case.


[Watch the Demo Video](demo.gif)



## Usage GPU enabled pytorch based	
The model was trained using pytorch and exported for CPU and GPU enabled, 

I have processed the output to be shown wether annotated or a message showing the center you can simply include the rostopic output with the message it needs for your application.

read yolo_gpu.py it helps you implement for your own application and ros version.

