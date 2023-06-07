# Convert the pre-trained ResNext101FF.pth to CoreML format

import torch
import cv2
import coremltools as ct
from detector.utils import build_model
from demo import Demo
from torch import nn
import json

LABELS = [
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "two_up",
    "two_up_inverted",
    "three2",
    "peace_inverted",
    "no_gesture"
]

threshold = 0.8

class WrappedModule(nn.Module):
    def __init__(self):
        super(WrappedModule, self).__init__()
        self.model = build_model("SSDLiteMobilenet_large", len(LABELS) + 1, checkpoint="SSDLite_MobilenetV3_large.pth", device="cpu").eval()
    
    def forward(self, x):
        output = self.model(x)[0]
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]
        return (scores, labels, boxes)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        ret, frame = cap.read()
        processed_frame, size, padded_size = Demo.preprocess(frame)
        traceable_model = WrappedModule().eval()
        trace = torch.jit.trace(traceable_model, processed_frame)
        _input = ct.ImageType(
            name="frame", 
            shape=processed_frame.shape
        )
        mlmodel = ct.convert(
            trace,
            inputs=[_input],
        )
        # Set the Model Metadata
        labels_json = {"labels": LABELS}

        mlmodel.type = 'imageSegmenter'
        mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)
        mlmodel.save("SSDLite_MobilenetV3_large.mlmodel")
        pass
    else:
        print("Cannot open camera")
        exit(0)