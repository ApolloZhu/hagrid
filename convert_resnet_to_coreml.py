# Convert the pre-trained ResNext101FF.pth to CoreML format

import torch
import cv2
import coremltools as ct
from classifier.utils import build_model
from demo_ff import Demo
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

class WrappedModule(nn.Module):
    def __init__(self):
        super(WrappedModule, self).__init__()
        self.model = build_model("ResNext101", len(LABELS) - 1, "cpu", checkpoint="ResNext101FF.pth", pretrained=True, freezed=True, ff=True).eval()
    
    def forward(self, x):
        res = self.model(x)
        x = res["gesture"]
        return x

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        processed_frame, size = Demo.preprocess(frame)
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
        mlmodel.save("ResNext101FF.mlmodel")
        pass
    else:
        print("Cannot open camera")
        exit(0)