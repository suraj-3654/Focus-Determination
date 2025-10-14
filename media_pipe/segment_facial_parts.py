"""
Imports for the segmentation class
"""
import os
import shutil
import glob
import cv2
import mediapipe as mp
from PIL import Image

from focus_model.focus_model import FocusModel
from focus_model.focus_model_onnx import FocusModelOnnx


class SegmentFacialParts:
    """
    Class contains the methods to segment the facial parts face, eyes, forehead.
    """
    _instance = None

    LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155]
    RIGHT_EYE = [362, 263, 387, 386, 385, 373, 380, 374, 381]
    FACE = list(range(0, 468))
    FOREHEAD = [10, 151, 9, 8, 107, 55, 65, 52, 53, 46, 70, 63, 105,
                66, 69, 108, 104, 103, 67, 109]

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
	    max_num_faces=20,  # Increase limit
	    refine_landmarks=True,
	    min_detection_confidence=0.3,  # Lower threshold
	    min_tracking_confidence=0.3
	    )
        self.output_dir = "segmented_images"
        self.obj_foucs_model = FocusModel()
        self.obj_foucs_model_onnx = FocusModelOnnx()


    def crop_part(self, image, landmarks, indices, padding=10):
        """
        Segments key facial regions such as eyes, forehead, and full face 
        from an input image using MediaPipe landmarks.
        """
        h, w, _ = image.shape
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x_coords, y_coords = zip(*points)
        x_min, x_max = max(min(x_coords) - padding, 0), min(max(x_coords) + padding, w)
        y_min, y_max = max(min(y_coords) - padding, 0), min(max(y_coords) + padding, h)
        return image[y_min:y_max, x_min:x_max]

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

        return image[head_top:head_bottom, head_left:head_right]

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


    def crop_forehead_adaptive(self, image, landmarks, indices):
        """
        Adaptive forehead cropping that works for different image sizes and subject scales
        """
        h, w, _ = image.shape

        # Calculate face scale and dimensions
        face_scale, face_width, face_height = self.get_face_scale(landmarks, image.shape)

        # Get forehead points
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x_coords, y_coords = zip(*points)

        # Get the topmost point for forehead top
        forehead_top = min(y_coords)
        forehead_left = min(x_coords)
        forehead_right = max(x_coords)

        # Find eyebrow landmarks to limit the bottom of forehead
        eyebrow_landmarks = [70, 63, 105, 66, 55, 65, 52, 53, 46, 107, 55, 8, 9, 10, 151]
        eyebrow_y_values = [int(landmarks[i].y * h)
                            for i in eyebrow_landmarks if i < len(landmarks)]
        eyebrow_y = min(eyebrow_y_values) if eyebrow_y_values else min(y_coords)

        # Calculate adaptive padding based on face scale
        # For larger faces (closer to camera), use smaller padding
        # For smaller faces (farther from camera), use larger padding
        base_padding = int(face_width * 0.1)  # 10% of face width
        min_padding = 10
        max_padding = 50

        adaptive_padding = max(min_padding, min(max_padding, base_padding))

        # Calculate adaptive extensions based on face scale - increased for complete forehead
        top_extension = int(face_height * 0.25)  # 25% of face height - more hair coverage
        bottom_extension = int(face_height * 0.25) #25% of face height-extend well into eyebrow area
        side_extension = int(face_width * 0.3)  # 30% of face width - more side coverage
        right_extension = int(face_width * 0.4)  # 40% of face width - more right side coverage

        # Define forehead boundaries with adaptive values
        y_min = max(forehead_top - adaptive_padding - top_extension, 0)
        y_max = max(eyebrow_y + bottom_extension, y_min + int(face_height * 0.3))
        x_min = max(forehead_left - adaptive_padding - side_extension, 0)
        x_max = min(forehead_right + adaptive_padding + right_extension, w)

        # Ensure we have a valid crop
        if y_max <= y_min:
            y_max = y_min + int(face_height * 0.3)

        # Additional safety check: allow much more area below eyebrows for complete forehead
        max_bottom = eyebrow_y + int(face_height * 0.35)  # Max 35% of face height below eyebrows
        y_max = min(y_max, max_bottom)

        return image[y_min:y_max, x_min:x_max]

    def crop_eyes_adaptive(self, image, landmarks, eye_indices):
        """
        Adaptive eye cropping based on face scale
        """
        h, w, _ = image.shape
        face_scale, face_width, face_height = self.get_face_scale(landmarks, image.shape)

        # Calculate adaptive padding for eyes
        eye_padding = int(face_width * 0.08)  # 8% of face width
        eye_padding = max(10, min(30, eye_padding))  # Between 10 and 30 pixels

        return self.crop_part(image, landmarks, eye_indices, padding=eye_padding)

    def process_folder(self, folder_path):
        """
        Process all images in a folder
        """
        print(f"=== Processing Folder: {folder_path} ===")

        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []

        for file in glob.iglob(os.path.join(folder_path, '*')):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(file)


        if not image_files:
            print(f"No image files found in {folder_path}")
            return

        print(f"Found {len(image_files)} image(s) to process")

        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)

        total_faces = 0
        processed_images = 0

        # Process each image
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            faces_processed = self.process_single_image(image_path, image_name)

            if faces_processed > 0:
                processed_images += 1
                total_faces += faces_processed

        # Summary
        print(f"\n=== Processing Complete ===")
        print(f"Processed {processed_images}/{len(image_files)} images")
        print(f"Total faces processed: {total_faces}")
        print(f"Output directory: {self.output_dir}")

        if processed_images == 0:
            print("No faces were detected in any images")
            print("Suggestions:")
            print("   - Check image quality and lighting")
            print("   - Ensure faces are clearly visible")
            print("   - Try different image formats")

    def process_single_image(self, image_path, image_name):
        """
        Process a single image and extract facial features
        """
        print(f"\n=== Processing: {image_name} ===")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return 0

        print(f"  Image size: {image.shape}")

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

        # Process each detected face
        for count, face_landmarks in enumerate(results.multi_face_landmarks):
            print(f"    Processing face {count + 1}/{face_count}...")

            try:
                landmarks = face_landmarks.landmark

                # Calculate face scale for debugging
                face_scale, face_width, face_height = self.get_face_scale(landmarks, image.shape)
                print(f"      Face scale: {face_scale:.3f}, Size: {face_width}x{face_height}")

                # Extract facial parts
                face_crop, face_info = self.crop_face_adaptive(image, landmarks, self.FACE)
                face_image = Image.fromarray(face_crop)
                face_image.save("temp_face.png")
                # face_score = self.obj_foucs_model.predict_score("temp_face.png")
                face_score = self.obj_foucs_model_onnx.make_prediction("temp_face.png")
                print(f"face_score:{face_score}, round:{face_score:.2f}")

                left_eye_crop = self.crop_eyes_adaptive(image, landmarks, self.LEFT_EYE)
                left_eye_image = Image.fromarray(left_eye_crop)
                left_eye_image.save("temp_le.png")
                # left_eye_score = self.obj_foucs_model.predict_score("temp_le.png")
                left_eye_score = self.obj_foucs_model_onnx.make_prediction("temp_le.png")
                print(f"left_eye_score:{left_eye_score}, round:{left_eye_score:.2f}")

                right_eye_crop = self.crop_eyes_adaptive(image, landmarks, self.RIGHT_EYE)
                right_eye_image = Image.fromarray(right_eye_crop)
                right_eye_image.save("temp_re.png")
                # right_eye_score = self.obj_foucs_model.predict_score("temp_re.png")
                right_eye_score = self.obj_foucs_model_onnx.make_prediction("temp_re.png")
                print(f"right_eye_score:{right_eye_score}, round:{right_eye_score:.2f}")

                head_crop = self.crop_head_adaptive(image, landmarks, self.FACE)
                head_image = Image.fromarray(head_crop)
                head_image.save("temp_head.png")
                # head_score = self.obj_foucs_model.predict_score("temp_head.png")
                head_score = self.obj_foucs_model_onnx.make_prediction("temp_head.png")
                print(f"head_score:{head_score}, round:{head_score:.2f}")

                # forehead_crop = self.crop_forehead_adaptive(image, landmarks, self.FOREHEAD)
                # Print face information as per requirements
                print(f"Face Center: ({face_info['center_x']}, {face_info['center_y']})")
                print(f"Face Size: {face_info['width']}x{face_info['height']}")
                print(f"Circumscribing Rectangle:({face_info['rect_left']},{face_info['rect_top']})"
                f"to ({face_info['rect_right']}, {face_info['rect_bottom']})"
                 )

                # Save with unique filenames
                cv2.imwrite(os.path.join(image_output_dir,
                        f"{face_score:.2f}_face_{count:02d}.png"), face_crop)
                cv2.imwrite(os.path.join(image_output_dir,
                        f"{left_eye_score:.2f}_left_eye_{count:02d}.png"), left_eye_crop)
                cv2.imwrite(os.path.join(image_output_dir,
                        f"{right_eye_score:.2f}_right_eye_{count:02d}.png"), right_eye_crop)
                # cv2.imwrite(os.path.join(image_output_dir,
                #                          f"forehead_{count:02d}.png"), forehead_crop)
                cv2.imwrite(os.path.join(image_output_dir,
                        f"{head_score:.2f}_head_{count:02d}.png"), head_crop)

                print(f"      ✅ Face {count + 1} processed successfully")
                successful_faces += 1

            except ValueError as e:
                print(f"Error processing face {count + 1}: {str(e)}")
                continue

        print(f"Output saved to: {image_output_dir}")
        print(f"Successfully processed: {successful_faces}/{face_count} faces")

        return successful_faces

    def clear_data(self):
        """This method deletes the segmented_image folder."""
        if self.output_dir and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print("Segmented output deleted.")
        else:
            print("No segmented output found.")
