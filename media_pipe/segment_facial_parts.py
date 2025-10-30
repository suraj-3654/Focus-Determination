"""
Imports for the segmentation class
"""
import os
import shutil
import glob
import tempfile
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from focus_model.focus_model_onnx import FocusModelOnnx
from utility.eye_segmentation_utils import EyeCalculator
from utility.update_json import JsonHandler

# ===== Constants (structure/readability, no behavior change) =====
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
DEFAULT_MAX_NUM_FACES = 20
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3
EYE_PADDING_RATIO = 0.08  # 8% of face width
EYE_PADDING_MIN = 10
EYE_PADDING_MAX = 30




class SegmentFacialParts:
    """
    Class contains the methods to segment facial parts (eyes, face, head).
    """
    _instance = None

    LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155]
    RIGHT_EYE = [362, 263, 387, 386, 385, 373, 380, 374, 381]
    FACE = list(range(0, 468))


    @classmethod
    def get_instance(cls):
        """
        Singleton class to provide the instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        constructor
        """
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=DEFAULT_MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.output_dir = None  # Will be set relative to input folder
        self.temp_dir = None  # Will be set when processing starts
        self.obj_foucs_model_onnx = FocusModelOnnx()
        self.obj_eye_calculator = EyeCalculator()
        self.obj_json_handler = JsonHandler()

    def format_data(self, face_number, info, label, focus_score):
        """
        Format information in the requested structure (works for both face and head)
        Args:   
            face_number: Face number for this face in the group
            info: info dict from crop_face_adaptive or crop_head_adaptive
            label: "h_face" or "h_head"
        Returns:
            dict: Formatted information
        """
        return {
            "face_number": face_number,
            "shapes": [
                {
                "label": label,
                "points": [
                    [info['rect_left'],info['rect_top']] , [info['rect_right'], info['rect_bottom']]
                    ],
                    "shape": "rectangle",   
                    "focus_score": focus_score
                }
                ],
            "shape": "rectangle",   
            "focus_score": None
        }

    def crop_part(self, image, landmarks, indices, padding=10):
        """
        Segments key facial regions such as eyes,head and full face 
        from an input image using MediaPipe landmarks.
        """
        h, w, _ = image.shape
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x_coords, y_coords = zip(*points)
        x_min, x_max = max(min(x_coords) - padding, 0), min(max(x_coords) + padding, w)
        y_min, y_max = max(min(y_coords) - padding, 0), min(max(y_coords) + padding, h)
        return image[y_min:y_max, x_min:x_max]

    def add_focus_score_annotation(self, image, focus_score, label="Focus"):
        """
        Add focus score annotation to image with proper padding
        Args:
            image: Input image (numpy array)
            focus_score: Focus score value to display
            label: Label text (e.g., "Left Eye", "Right Eye", "Face", "Head")
        Returns:
            Annotated image with padding and text
        """
        # Get image dimensions
        height, width = image.shape[:2]

        # Calculate font scale based on image size - increase for better visibility
        font_scale = min(width, height) / 300.0  # Increased from 500.0 to 300.0
        font_scale = max(0.4, min(1.2, font_scale))  # Increased max from 0.8 to 1.2, min from 0.25 to 0.4

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = max(1, int(font_scale * 1.5))  # Reduced back to moderate boldness

        # Text to display
        text = f"{label}: {focus_score:.2f}"

        # Get text size for calculating padding
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate padding based on text size and image size - be VERY generous
        # Ensure padding is at least 50% of text height, but not less than 30 pixels
        text_padding = max(30, int(text_height * 0.5))  # Increased from 20 to 30, from 0.3 to 0.5
        side_padding = max(10, int(text_width * 0.1))  # Reduced from 25 to 10, from 0.4 to 0.1 for better left alignment

        # Calculate new image dimensions with padding - ensure width accommodates text
        new_height = height + text_padding + text_height + text_padding
        new_width = max(width, text_width + (side_padding * 2))  # Ensure width fits text

        # Create new image with padding (white background)
        padded_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

        # Place original image in the center-bottom of padded image
        start_y = text_padding + text_height + text_padding
        start_x = (new_width - width) // 2  # Center horizontally
        padded_image[start_y:start_y + height, start_x:start_x + width] = image

        # Position text in the top padding area - COMPLETELY LEFT ALIGNED
        text_x = 5  # Minimal padding from left edge for true left alignment
        text_y = text_padding + text_height

        # Draw small black rectangle only around the text
        cv2.rectangle(padded_image, 
                     (text_x - 5, text_y - text_height - 5),  # Small padding around text
                     (text_x + text_width + 5, text_y + baseline + 5),  # Small padding around text
                     (0, 0, 0), -1)  # Black background

        # Draw text
        cv2.putText(padded_image, text, (text_x, text_y), 
                   font, font_scale, (255, 255, 255), thickness)  # White text

        return padded_image

    def crop_head_adaptive(self, image, landmarks, face_indices):
        """
        Extract the entire head area (1.3x face height) as per requirements
        """
        h, w, _ = image.shape

        # Calculate face scale and dimensions
        face_scale, face_width, face_height = self.get_face_scale(landmarks, image.shape)

        # Get face bounding box from all face landmarks
        face_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in face_indices]
        face_x_coords, face_y_coords = zip(*face_points)

        # Calculate face center and dimensions
        face_center_x = (min(face_x_coords) + max(face_x_coords)) // 2
        face_center_y = (min(face_y_coords) + max(face_y_coords)) // 2

        # Calculate head dimensions (1.3x face height as per requirements)
        head_height = int(face_height * 1.3)
        head_width = int(face_width * 1.1)  # Slightly wider than face for better head coverage

        # Calculate adaptive padding based on face scale
        base_padding = int(face_width * 0.05)  # 5% of face width
        min_padding = 5
        max_padding = 25
        adaptive_padding = max(min_padding, min(max_padding, base_padding))

        # Calculate head boundaries centered on face center
        head_left = max(face_center_x - head_width // 2 - adaptive_padding, 0)
        head_right = min(face_center_x + head_width // 2 + adaptive_padding, w)
        head_top = max(face_center_y - head_height // 2 - adaptive_padding, 0)
        head_bottom = min(face_center_y + head_height // 2 + adaptive_padding, h)

        # Ensure valid dimensions
        if head_bottom <= head_top:
            head_bottom = head_top + head_height
        if head_right <= head_left:
            head_right = head_left + head_width

        # Store head information for potential future use
        head_info = {
            'center_x': face_center_x,
            'center_y': face_center_y,
            'width': head_width,
            'height': head_height,
            'rect_left': head_left,
            'rect_top': head_top,
            'rect_right': head_right,
            'rect_bottom': head_bottom
        }

        return image[head_top:head_bottom, head_left:head_right], head_info

    def get_face_scale(self, landmarks, image_shape):
        """
        Calculate the scale of the face relative to image size
        """
        h, w = image_shape[:2]

        # Get face bounding box
        face_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in range(468)]
        x_coords, y_coords = zip(*face_points)

        face_width = max(x_coords) - min(x_coords)
        face_height = max(y_coords) - min(y_coords)

        # Calculate scale factors
        width_scale = face_width / w
        height_scale = face_height / h

        return min(width_scale, height_scale), face_width, face_height

    def crop_face_adaptive(self, image, landmarks, face_indices):
        """
        Extract face information as per requirements:
        - Center coordinates
        - Circumscribing rectangle (half face size from center in all directions)
        """
        h, w, _ = image.shape

        # Calculate face scale and dimensions
        face_scale, face_width, face_height = self.get_face_scale(landmarks, image.shape)

        # Get face bounding box from all face landmarks
        face_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in face_indices]
        face_x_coords, face_y_coords = zip(*face_points)

        # Calculate face center coordinates (as per requirements)
        face_center_x = (min(face_x_coords) + max(face_x_coords)) // 2
        face_center_y = (min(face_y_coords) + max(face_y_coords)) // 2

        # Calculate circumscribing rectangle (half face size from center)
        half_face_width = face_width // 2
        half_face_height = face_height // 2

        # Calculate rectangle boundaries
        rect_left = max(face_center_x - half_face_width, 0)
        rect_right = min(face_center_x + half_face_width, w)
        rect_top = max(face_center_y - half_face_height, 0)
        rect_bottom = min(face_center_y + half_face_height, h)

        # Ensure valid dimensions
        if rect_bottom <= rect_top:
            rect_bottom = rect_top + face_height
        if rect_right <= rect_left:
            rect_right = rect_left + face_width

        # Store face information for potential future use
        face_info = {
            'center_x': face_center_x,
            'center_y': face_center_y,
            'width': face_width,
            'height': face_height,
            'rect_left': rect_left,
            'rect_top': rect_top,
            'rect_right': rect_right,
            'rect_bottom': rect_bottom
        }

        return image[rect_top:rect_bottom, rect_left:rect_right], face_info

    def crop_eyes_adaptive(self, image, landmarks, eye_indices):
        """
        Adaptive eye cropping based on face scale
        """
        h, w, _ = image.shape
        face_scale, face_width, face_height = self.get_face_scale(landmarks, image.shape)

        # Calculate adaptive padding for eyes
        eye_padding = int(face_width * EYE_PADDING_RATIO)
        eye_padding = max(EYE_PADDING_MIN, min(EYE_PADDING_MAX, eye_padding))

        return self.crop_part(image, landmarks, eye_indices, padding=eye_padding)

    def process_folder(self, folder_path):
        """
        Process all images in a folder
        """
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return

        # Get all image files
        image_files = []
        for file in glob.iglob(os.path.join(folder_path, '*')):
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append(file)


        if not image_files:
            print(f"No image files found in {folder_path}")
            return

        print(f"Found {len(image_files)} image(s) to process")

        # Create output directory relative to input folder
        # segmented_images will be created in the same directory as the input folder
        input_folder_parent = os.path.dirname(os.path.abspath(folder_path))
        self.output_dir = os.path.join(input_folder_parent, "segmented_images")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create temp directory for temporary image files
        self.temp_dir = tempfile.mkdtemp(prefix="focus_annotation_")

        total_faces = 0
        processed_images = 0

        # Process each image
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            faces_processed = self.process_image(image_path, image_name)

            if faces_processed > 0:
                processed_images += 1
                total_faces += faces_processed

        # Clean up temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                pass

        # Summary
        print(f"Processed {processed_images}/{len(image_files)} images, {total_faces} faces")
        print(f"Output directory: {self.output_dir}")

        if processed_images == 0:
            print("No faces detected in any images")

    def process_image(self, image_path, image_name):
        """
        Process a single image and extract facial features
        """
        print(f"Processing: {image_name}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return 0
        
        # Get image dimensions for JSON metadata
        image_height, image_width = image.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print(f"No faces detected in {image_name}")
            return 0

        face_count = len(results.multi_face_landmarks)
        print(f"Detected {face_count} face(s)")

        # Create image-specific output directory
        image_output_dir = os.path.join(self.output_dir, os.path.splitext(image_name)[0])
        os.makedirs(image_output_dir, exist_ok=True)

        successful_faces = 0
        try:
            json_eye_dict = {}
            focus_scores = {}
            eye_info = self.obj_eye_calculator.demonstrate_eye_calculator(image_path)
            
            # Create a fresh JSON handler for this image
            from utility.update_json import JsonHandler
            image_json_handler = JsonHandler()
            face_data = eye_info['faces_data']
            for face_num, face in enumerate(face_data):
                left_eye_crop = face['left_eye_crop']
                rgb_left_eye_crop = cv2.cvtColor(left_eye_crop, cv2.COLOR_BGR2RGB)
                left_eye_image = Image.fromarray(rgb_left_eye_crop)
                temp_le_path = os.path.join(self.temp_dir, f"temp_le_{face_num}.png")
                left_eye_image.save(temp_le_path)
                left_eye_score = self.obj_foucs_model_onnx.make_prediction(temp_le_path)

                right_eye_crop = face['right_eye_crop']
                rgb_right_eye_crop = cv2.cvtColor(right_eye_crop, cv2.COLOR_BGR2RGB)
                right_eye_image = Image.fromarray(rgb_right_eye_crop)
                temp_re_path = os.path.join(self.temp_dir, f"temp_re_{face_num}.png")
                right_eye_image.save(temp_re_path)
                right_eye_score = self.obj_foucs_model_onnx.make_prediction(temp_re_path)
                
                # Add annotations and save eye images
                left_eye_annotated = self.add_focus_score_annotation(left_eye_crop, left_eye_score, "Left Eye")
                right_eye_annotated = self.add_focus_score_annotation(right_eye_crop, right_eye_score, "Right Eye")
                
                # Get filename without extension
                filename_base = os.path.splitext(image_name)[0]
                
                # Get eye rectangle coordinates (tuple format: top, bottom, left, right)
                left_eye_rect = face['left_eye_rectangle']
                right_eye_rect = face['right_eye_rectangle']
                
                # Calculate width and height from rectangle coordinates
                left_eye_width = left_eye_rect[3] - left_eye_rect[2]  # right - left
                left_eye_height = left_eye_rect[1] - left_eye_rect[0]  # bottom - top
                right_eye_width = right_eye_rect[3] - right_eye_rect[2]  # right - left
                right_eye_height = right_eye_rect[1] - right_eye_rect[0]  # bottom - top
                
                # Save with new naming format using eye coordinates
                cv2.imwrite(os.path.join(image_output_dir,
                                f"{filename_base}_{left_eye_rect[0]}_{left_eye_rect[2]}_{left_eye_width}_{left_eye_height}_{left_eye_score:.2f}_l_{face['face_number']:02d}.png"), left_eye_annotated)
                cv2.imwrite(os.path.join(image_output_dir,
                            f"{filename_base}_{right_eye_rect[0]}_{right_eye_rect[2]}_{right_eye_width}_{right_eye_height}_{right_eye_score:.2f}_r_{face['face_number']:02d}.png"), right_eye_annotated)
                focus_scores[face_num] = {"left_eye_score": left_eye_score}
                focus_scores[face_num]["right_eye_score"] = right_eye_score
            face_count = eye_info["total_faces"]
            for face_num in range(face_count):
                json_left_eye_info = eye_info["json_eye_info"][face_num]["left_eye"]["shapes"]

                # Iterate through shapes list to add focus_score
                for shape in json_left_eye_info:
                    shape["focus_score"] = focus_scores[face_num]["left_eye_score"]

                json_right_eye_info = eye_info["json_eye_info"][face_num]["right_eye"]["shapes"]

                # Iterate through shapes list to add focus_score
                for shape in json_right_eye_info:
                    shape["focus_score"] = focus_scores[face_num]["right_eye_score"]

                json_eye_dict[face_num] = json_left_eye_info + json_right_eye_info

            # Process each detected face
            for count, face_landmarks in enumerate(results.multi_face_landmarks):
                landmarks = face_landmarks.landmark
                # Extract facial parts
                face_crop, face_info = self.crop_face_adaptive(image, landmarks, self.FACE)

                rgb_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_image = Image.fromarray(rgb_face_crop)
                temp_face_path = os.path.join(self.temp_dir, f"temp_face_{count}.png")
                face_image.save(temp_face_path)
                face_score = self.obj_foucs_model_onnx.make_prediction(temp_face_path)
                json_face_data = self.format_data(count, face_info, "h_face", face_score)
                head_crop, head_info = self.crop_head_adaptive(image, landmarks, self.FACE)

                rgb_head_crop = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
                head_image = Image.fromarray(rgb_head_crop)
                temp_head_path = os.path.join(self.temp_dir, f"temp_head_{count}.png")
                head_image.save(temp_head_path)
                head_score = self.obj_foucs_model_onnx.make_prediction(temp_head_path)
                json_head_data = self.format_data(count, head_info, "h_head", head_score)

                # Add annotations and save face/head images
                face_annotated = self.add_focus_score_annotation(face_crop, face_score, "Face")
                head_annotated = self.add_focus_score_annotation(head_crop, head_score, "Head")

                # Get filename without extension
                filename_base = os.path.splitext(image_name)[0]

                # Save with new naming format
                cv2.imwrite(os.path.join(image_output_dir,
                        f"{filename_base}_{face_info['rect_top']}_{face_info['rect_left']}_{face_info['width']}_{face_info['height']}_{face_score:.2f}_f_{count:02d}.png"), face_annotated)
                cv2.imwrite(os.path.join(image_output_dir,
                        f"{filename_base}_{head_info['rect_top']}_{head_info['rect_left']}_{head_info['width']}_{head_info['height']}_{head_score:.2f}_h_{count:02d}.png"), head_annotated)
                successful_faces += 1
                json_dict = {}
                json_dict["shapes"] = json_eye_dict[count]
                json_dict["shapes"].extend(json_face_data["shapes"] + json_head_data["shapes"])

                # Prepare image metadata for JSON
                image_info = {
                    "imagePath": image_path,
                    "imageHeight": image_height,
                    "imageWidth": image_width
                }

                image_json_handler.update_json(json_dict, output_path=image_output_dir, image_info=image_info)
            print(f"Saved to: {image_output_dir}, {successful_faces}/{face_count} faces")
        except ValueError as e:
            print(f"Error processing face {count + 1}: {str(e)}")
        return successful_faces

    def clear_data(self):
        """This method deletes the segmented_image folder."""
        if self.output_dir and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # Also clean up temp directory if it exists
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                pass
