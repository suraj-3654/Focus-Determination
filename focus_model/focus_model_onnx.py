"""
Imports for the focus mode onnx file.
"""
import os
import sys
import onnxruntime as ort
import numpy as np
import cv2

try:
    from utility.resource_path import resource_path
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utility.resource_path import resource_path

class FocusModelOnnx():
    """This class contains the methods for loading and make prdictions 
     with focus determination model with onnx runtime
    """

    def __init__(self):
        model_path = resource_path("model/focusmodel.onnx")
        self.model = model_path
        self.session = ort.InferenceSession(self.model)
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def normalize(self, input):
        """
        Applies necessary standardization/normalization to the input tensor.

        Args:
            input_tensor (np.ndarray): 4D float32 tensor (Batch, H, W, C).
            
        Returns:
            np.ndarray: The normalized float32 tensor (e.g., mean-subtracted).
        """
        input = input.astype(np.float32)
        input/=255.
        input-=np.array([0.485, 0.456, 0.406])
        input/=np.array([0.229, 0.224, 0.225])
        return input

    def make_prediction(self, img):
        """
         This method is to make prediction with focus determination onnx model

         Args:
            Path of the image to calculate the focus score
         Returns:

            focus score of the image
        """
        session = ort.InferenceSession(self.model)
        input = cv2.imread(img)[:,:,::-1]
        input = cv2.resize(input,(128,128),interpolation=cv2.INTER_LINEAR)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        input_with_batch = np.expand_dims(input, axis=0)
        if 'keras' not in os.path.basename(self.model):
            input = self.normalize(input_with_batch)
        else:
            input = input_with_batch.astype(np.float32) / 255.0
        predictions = session.run(output_names, {input_name: input})
        focus_score = predictions[0][0][0]
        return focus_score
