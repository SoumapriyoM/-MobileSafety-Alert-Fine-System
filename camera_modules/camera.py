import cv2
import numpy as np

class webcam:
    def __init__(self, fps=30):
        self.video = cv2.VideoCapture(0)
        self.frame_time = int(1000 / fps)

        # Load the Haar cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load the MobileNet SSD model for mobile phone detection
        prototxt = "camera_modules/MobileNetSSD_deploy.prototxt"
        caffe_model = "camera_modules/MobileNetSSD_deploy.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

    def __del__(self):
        self.video.release()

    def detect_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return faces

    def detect_mobiles(self, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        mobile_rectangles = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if confidence > 0.5:  # Class ID for mobile phone
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                mobile_rectangles.append((startX, startY, endX - startX, endY - startY))

        return mobile_rectangles

    def draw_rectangles(self, frame, rectangles, color):
        for (x, y, w, h) in rectangles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        return frame

    def get_frame(self):
        success, frame = self.video.read()
        cv2.waitKey(self.frame_time)

        # Detect faces
        faces = self.detect_faces(frame)

        # Detect mobile phones
        mobiles = self.detect_mobiles(frame)

        # Draw rectangles around faces
        frame_with_faces = self.draw_rectangles(frame.copy(), faces, (0, 255, 0))

        # Draw rectangles around mobile phones
        frame_with_objects = self.draw_rectangles(frame_with_faces, mobiles, (0, 0, 255))

        # Flip the image horizontally for correct visualization
        frame_with_objects = cv2.flip(frame_with_objects, 1)

        # Encode the frame to JPEG format
        _, jpeg = cv2.imencode(".jpg", frame_with_objects)

        return jpeg.tobytes()