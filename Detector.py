import cv2
import numpy as np

class Detector:
    def __init__(self, configPath, modelPath, classesPath):
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Load YOLO
        self.net = cv2.dnn.readNet(self.modelPath, self.configPath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
        print("Classes loaded:", self.classesList)

    def predict(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        layerNames = self.net.getLayerNames()
        outputLayers = [layerNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

        detections = self.net.forward(outputLayers)

        boxes = []
        confidences = []
        classIDs = []

        for output in detections:
            print(f"Detection Output Shape: {output.shape}")
            for detection in output:
                detection = detection.reshape(-1)
                print(f"Detection: {detection.shape}")
                for i in range(0, len(detection), 85):  # YOLOv3 produces 85 outputs per object
                    obj = detection[i:i+85]
                    scores = obj[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > 0.5:
                        centerX, centerY, w, h = obj[:4]
                        centerX *= width
                        centerY *= height
                        w *= width
                        h *= height

                        x = int(centerX - w / 2)
                        y = int(centerY - h / 2)

                        boxes.append([x, y, int(w), int(h)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        return [(boxes[i[0]], confidences[i[0]], classIDs[i[0]]) for i in indices]

    def detectObjects(self, videoSource):
        cap = cv2.VideoCapture(videoSource)
        if not cap.isOpened():
            print("Error opening video source")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            predictions = self.predict(frame)

            for (box, confidence, classID) in predictions:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{self.classesList[classID]}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
