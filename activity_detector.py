import cv2
import numpy as np
import onnxruntime as ort

class WorkerActivityDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame):
        resized = cv2.resize(frame, (224, 224))
        normalized = resized.astype('float32') / 255.0
        blob = np.transpose(normalized, (2, 0, 1))[None, ...]
        return blob

    def detect(self, frame):
        input_blob = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_blob})
        result = outputs[0]
        return result[0][0] > 0.5
