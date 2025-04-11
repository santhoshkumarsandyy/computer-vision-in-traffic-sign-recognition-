import os
from Detector import Detector

def main():
    print("Starting real-time object detection")

    videoSource = 0  # Use 0 for webcam

    configPath = os.path.join("model_data", "yolov3", "yolov3.cfg")
    modelPath = os.path.join("model_data", "yolov3", "yolov3.weights")
    classesPath = os.path.join("model_data", "yolov3", "coco.names")

    print(f"Config Path: {configPath}")
    print(f"Model Path: {modelPath}")
    print(f"Classes Path: {classesPath}")

    detector = Detector(configPath=configPath, modelPath=modelPath, classesPath=classesPath)
    detector.detectObjects(videoSource)

if __name__ == '__main__':
    main()
