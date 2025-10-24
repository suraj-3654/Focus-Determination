"""
Imports for focus_determination model.
"""

import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image


# class FocusModel():
#     """
#     This class contains the methods for loading and 
#     make prdictions with focus determination model.
#     """
#     def __init__(self):
#         """
#         constructor call
#         """
#         self.model_name = "./model/focusmodel.h5"
#         self.model = tf.keras.models.load_model(self.model_name)

#     def normalize(self, input):
#         """
#         Applies necessary standardization/normalization to the input tensor.

#         Args:
#             input_tensor (np.ndarray): 4D float32 tensor (Batch, H, W, C).
            
#         Returns:
#             np.ndarray: The normalized float32 tensor (e.g., mean-subtracted).
#         """
#         input = input.astype(np.float32)
#         input/=255.
#         input-=np.array([0.485, 0.456, 0.406])
#         input/=np.array([0.229, 0.224, 0.225])

#     def predict_score(self, img):
#         """
#         This method is to predict the focus score with the input image.
#         """
#         # print("Inside_model")
#         # print(img)
        
#         input = cv2.imread("./data/single_image/New folder/face_01.jpg")[:,:,::-1]
#         input = cv2.resize(input,(128,128),interpolation=cv2.INTER_LINEAR)
#         # h5 from keras requires input_range [0 ~255]
#         # h5 from pytorch requires normalization
#         if 'keras' not in os.path.basename(self.model_name):
#             input = self.normalize(input)
#         output = (self.model(input[None],training=False))
#         numpy_result = output.numpy()
#         focus_score = numpy_result[0, 0]
#         return focus_score
class FocusModel():

    def __init__(self):
        """
        constructor call
        """
        self.model_name = "./model/focusmodel.h5"
        self.model = tf.keras.models.load_model(self.model_name)

    def norm(self, input):
        input = input.astype(np.float32)
        input/=255.
        input-=np.array([0.485, 0.456, 0.406])
        input/=np.array([0.229, 0.224, 0.225])
        return input

    def  predict_score(self, img):
        input = cv2.imread(img)[:,:,::-1]
        input = cv2.resize(input,(128,128),interpolation=cv2.INTER_LINEAR)
        # h5 from keras requires input_range [0 ~255]
        # h5 from pytorch requires normalization
        if 'keras' not in os.path.basename(self.model_name):
            input = self.norm(input)
        output = (self.model(input[None],training=False))
        numpy_array = output.numpy()
        focus_score = numpy_array[0, 0]
        print(f"{img}:{focus_score}")
        print(focus_score)
        return focus_score




