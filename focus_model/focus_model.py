"""
Imports for focus_determination model.
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class FocusModel():
    """
    This class contains the methods for loading and 
    make prdictions with focus determination model.
    """
    def __init__(self):
        """
        constructor call
        """
        self.model = load_model("./model/focusmodel.h5")

    def predict_score(self, img):
        """
        This method is to predict the focus score with the input image.
        """
        # print("Inside_model")
        # print(img)
        
        img = image.load_img(img, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        input_data = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(input_data)
        # print(predictions)
        focus_score = predictions[0][0]  # Extract the float value
        # print(f"Focus Score: {focus_score:.2f}")
        return focus_score



