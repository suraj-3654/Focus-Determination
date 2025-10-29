"""
Eye Calculator Module
Implements precise eye calculation with 1/3 distance formula and magnification
"""
import json
import os
import cv2
import mediapipe as mp
import numpy as np

# Eye landmark indices
LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 373, 380, 374, 381]

class EyeCalculator:
    """
    Dedicated eye calculation class with precise 1/3 distance formula
    """
    def __init__(self, config_path="./utility/config.json"):
        """
        Initialize eye calculator with configuration
        Args:
            config_path (str): Path to eye configuration file
        """
        self.config = self._load_config(config_path)
        self.eye_mag_factor = self.config.get("eye_magnification", 1.0)

        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=20,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    def _load_config(self, config_path):
        """
        Load eye configuration from external file
        """
        default_config = {"eye_magnification": 1.0}

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded eye configuration from {config_path}")
                return config
            except Exception as e:
                print(f"❌ Error loading eye config: {e}")
                print("Using default eye configuration")
        else:
            print(f"❌ Eye config file {config_path} not found, using default")
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"✅ Created default eye config file: {config_path}")

        return default_config

    def _get_eye_centers(self, landmarks, image_shape):
        """
        Calculate center coordinates of left and right eyes
        Args:
            landmarks: MediaPipe landmarks
            image_shape: Image shape (height, width, channels)
        Returns:
            tuple: ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y))
        """
        h, w = image_shape[:2]

        # Left eye center calculation
        left_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        left_eye_x = sum(point[0] for point in left_eye_points) // len(left_eye_points)
        left_eye_y = sum(point[1] for point in left_eye_points) // len(left_eye_points)

        # Right eye center calculation
        right_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
        right_eye_x = sum(point[0] for point in right_eye_points) // len(right_eye_points)
        right_eye_y = sum(point[1] for point in right_eye_points) // len(right_eye_points)

        return (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)

    def _calculate_eye_distance(self, left_eye_center, right_eye_center):
        """
        Calculate distance between eye centers using Pythagorean theorem
        Args:
            left_eye_center: (x, y) coordinates of left eye
            right_eye_center: (x, y) coordinates of right eye
        Returns:
            float: Euclidean distance between eye centers
        """
        # Pythagorean theorem: c = √(a² + b²)
        eye_distance = ((right_eye_center[0] - left_eye_center[0])**2 +
                       (right_eye_center[1] - left_eye_center[1])**2)**0.5 
        return eye_distance

    def _calculate_eye_size(self, left_eye_center, right_eye_center):
        """
        Calculate eye size using the special 1/3 distance formula
        Args:
            left_eye_center: (x, y) coordinates of left eye
            right_eye_center: (x, y) coordinates of right eye
        Returns:
            tuple: (base_eye_size, final_eye_size, half_final_size)
        """
        # Calculate distance between eye centers
        eye_distance = self._calculate_eye_distance(left_eye_center, right_eye_center)

        # Base eye size = 1/3 of distance between eyes (SPECIAL FORMULA)
        base_eye_size = eye_distance / 3

        # Final size with magnification factor
        final_eye_size = base_eye_size * self.eye_mag_factor

        # Half size for circumscribing rectangle
        half_final_size = final_eye_size / 2

        return base_eye_size, final_eye_size, half_final_size

    def extract_eye_crop(self, image_array, eye_center, other_eye_center):
        """
        Extract eye area using precise 1/3 distance calculation
        Args:
            image_array (np.ndarray): The input image (H, W, C)
            eye_center (tuple): (x, y) coordinates of the eye to be extracted
            other_eye_center (tuple): (x, y) coordinates of the other eye
        Returns:
            np.ndarray: Cropped eye image
            tuple: (top, bottom, left, right) coordinates of the crop
        """
        h, w = image_array.shape[:2]

        # Calculate eye size using special formula
        base_eye_size, final_eye_size, half_final_size = self._calculate_eye_size(eye_center, other_eye_center)

        # Calculate circumscribing rectangle boundaries
        eye_x, eye_y = eye_center

        # Rectangle extends half_final_size from center
        left = max(int(eye_x - half_final_size), 0)
        right = min(int(eye_x + half_final_size), w)
        top = max(int(eye_y - half_final_size), 0)
        bottom = min(int(eye_y + half_final_size), h)

        # Extract crop
        eye_crop = image_array[top:bottom, left:right]

        return eye_crop, (top, bottom, left, right)

    def format_data(self, face_number, eye_rect, label):
        """
        Format eye information in the requested structure
        Args:
            face_number: Face number for this face in the group
            eye_rect: (top, bottom, left, right) tuple
            label: "h_l_eye" or "h_r_eye"
        Returns:
            dict: Formatted eye information
        """
        return {

            "face_number": face_number,
            "shapes": [
                {
                "label": label,
                "points": [
                    [eye_rect[2],eye_rect[0]] , [eye_rect[3], eye_rect[1]]
                    ],
                "shape": "rectangle",   
                "focus_score": None
                }
                ]
        }

    def calculate_eye_information(self, landmarks, image_shape):
        """
        Calculate complete eye information using special 1/3 distance formula
        Args:
            landmarks: MediaPipe landmarks
            image_shape: Image shape
        Returns:
            dict: Complete eye information
        """
        # Get eye centers
        left_eye_center, right_eye_center = self._get_eye_centers(landmarks, image_shape)

        # Calculate eye size using special formula
        base_eye_size, final_eye_size, half_final_size = self._calculate_eye_size(left_eye_center,
                                                                                  right_eye_center)

        # Calculate eye distance for verification
        eye_distance = self._calculate_eye_distance(left_eye_center, right_eye_center)

        return {
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center,
            'eye_distance': eye_distance,
            'base_eye_size': base_eye_size,
            'final_eye_size': final_eye_size,
            'half_final_size': half_final_size,
            'eye_magnification': self.eye_mag_factor
        }

    def process_image_eyes(self, image_path, save_eyes=True, output_dir="eye_output"):
        """
        Process image and extract eye information for ALL faces in group images
        Args:
            image_path (str): Path to the input image
            save_eyes (bool): Whether to save eye crops as RGB images
            output_dir (str): Directory to save eye images
        Returns:
            dict: Eye information and crops for all faces
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        # Process ALL detected faces for group images
        all_faces_data = []
        faces_data = []
        face_count = len(results.multi_face_landmarks)

        print(f"✅ Detected {face_count} face(s) in the image")

        for count, face_landmarks in enumerate(results.multi_face_landmarks):
            print(f"    Processing face {count + 1}/{face_count}...")

            try:
                # Get landmarks for current face
                landmarks = face_landmarks.landmark

                # Calculate eye information for this face
                eye_info = self.calculate_eye_information(landmarks, image.shape)

                # Extract eye crops using special calculation for this face
                left_eye_crop, left_eye_rect = self.extract_eye_crop(image,
                                        eye_info['left_eye_center'], eye_info['right_eye_center'])
                right_eye_crop, right_eye_rect = self.extract_eye_crop(image,
                                        eye_info['right_eye_center'], eye_info['left_eye_center'])
                # Collect formatted eye data
                left_eye_formatted = self.format_data(count, left_eye_rect, "h_l_eye")
                right_eye_formatted = self.format_data(count, right_eye_rect, "h_r_eye")
                
                faces_data.append({
                    "face_number": count,
                    "left_eye": left_eye_formatted,
                    "right_eye": right_eye_formatted
                })

                # Store data for this face
                face_data = {
                    'face_number': count,
                    'left_eye_crop': left_eye_crop,
                    'right_eye_crop': right_eye_crop,
                    'left_eye_rectangle': left_eye_rect,
                    'right_eye_rectangle': right_eye_rect,
                    **eye_info
                }

                all_faces_data.append(face_data)

                print(f"      ✅ Face {count + 1} processed successfully")
                print(f"         Left Eye Center: {eye_info['left_eye_center']}")
                print(f"         Right Eye Center: {eye_info['right_eye_center']}")
                print(f"         Eye Distance: {eye_info['eye_distance']:.2f} pixels")
                print(f"         Base Eye Size: {eye_info['base_eye_size']:.2f} pixels")

            except Exception as e:
                print(f"      ❌ Error processing face {count + 1}: {str(e)}")
                continue

        # Return data for all faces
        return {
            'total_faces': face_count,
            'successful_faces': len(all_faces_data),
            'faces_data': all_faces_data,
            'json_eye_info': faces_data
        }

    def demonstrate_eye_calculator(self, image_path,
                    config_path="./utility/config.json", save_eyes=True, output_dir="eye_output"):
        """
        Demonstrate the eye calculator functionality for multiple faces
        """
        print(f"\n=== Eye Calculator Demonstration (Multi-Face) ===")
        print(f"Image: {image_path}")
        print(f"Configuration: {config_path}")
        print(f"Save Eyes: {save_eyes}")
        print(f"Output Directory: {output_dir}")

        # Initialize eye calculator
        eye_calc = EyeCalculator(config_path=config_path)

        # Process image with eye saving for all faces
        result = eye_calc.process_image_eyes(image_path, save_eyes=save_eyes, output_dir=output_dir)

        if result is None:
            print("❌ No face detected or image could not be loaded")
            return

        print(f"\n✅ Processing completed!")
        print(f"📊 SUMMARY:")
        print(f"   Total Faces Detected: {result['total_faces']}")
        print(f"   Successfully Processed: {result['successful_faces']}")

        # Display results for each face
        for face_data in result['faces_data']:
            face_num = face_data['face_number']
            print(f"\n👁️ FACE {face_num + 1} EYE CALCULATION RESULTS:")
            print(f"   Left Eye Center: {face_data['left_eye_center']}")
            print(f"   Right Eye Center: {face_data['right_eye_center']}")
            print(f"   Eye Distance: {face_data['eye_distance']:.2f} pixels")
            print(f"   Base Eye Size: {face_data['base_eye_size']:.2f} pixels (1/3 of distance)")
            print(f"   Final Eye Size: {face_data['final_eye_size']:.2f} pixels (with magnification)")
            print(f"   Half Final Size: {face_data['half_final_size']:.2f} pixels")
            print(f"   Eye Magnification: {face_data['eye_magnification']}")
            print(f"   Left Eye Rectangle: {face_data['left_eye_rectangle']}")
            print(f"   Right Eye Rectangle: {face_data['right_eye_rectangle']}")
            print(f"   Left Eye Crop Shape: {face_data['left_eye_crop'].shape}")
            print(f"   Right Eye Crop Shape: {face_data['right_eye_crop'].shape}")

        return result
