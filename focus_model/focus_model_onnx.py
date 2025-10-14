"""
Imports for the focus mode onnx file.
"""
import onnxruntime as ort
import numpy as np
from tensorflow.keras.preprocessing import image


class FocusModelOnnx():
    """This class contains the methods for loading and make prdictions 
    with focus determination model with onnx runtime"""

    def __init__(self):
        self.session = ort.InferenceSession("./model/focusmodel.onnx")
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    # Load and preprocess the image

    def make_prediction(self,img):
        """
        This method is to make prediction with focus determination onnx model
        """
        img = image.load_img(img, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
        # Run inference
        result = self.session.run([self.output_name], {self.input_name: input_data})
        print(result)
        score = result[0][0][0] # Extract scalar value
        return score
