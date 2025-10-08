import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

image = cv2.imread("./data/face_02.jpg")
if image is None:
    print("Image not found.")
    exit()
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

# Indices based on MediaPipe's 468-point model
LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 373, 380, 374, 381]
FACE = list(range(0, 468))  # Use convex hull for full face
# Forehead region: only the top part of the face (forehead area)
FOREHEAD = [10, 151, 9, 8, 107, 55, 65, 52, 53, 46, 70, 63, 105, 66, 69, 108, 104, 103, 67, 109]

def crop_part(image, landmarks, indices, padding=10):
    h, w, _ = image.shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*points)
    x_min, x_max = max(min(x_coords) - padding, 0), min(max(x_coords) + padding, w)
    y_min, y_max = max(min(y_coords) - padding, 0), min(max(y_coords) + padding, h)
    return image[y_min:y_max, x_min:x_max]

def crop_forehead(image, landmarks, indices, padding=20):
    """Special function for forehead cropping - only the forehead area"""
    h, w, _ = image.shape
    
    # Get forehead points
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_coords, y_coords = zip(*points)
    
    # Get the topmost point for forehead top
    forehead_top = min(y_coords)
    forehead_left = min(x_coords)
    forehead_right = max(x_coords)
    
    # Find eyebrow landmarks to limit the bottom of forehead
    # Use multiple eyebrow landmarks for better accuracy
    eyebrow_landmarks = [70, 63, 105, 66, 55, 65, 52, 53, 46, 107, 55, 8, 9, 10, 151]
    eyebrow_y_values = [int(landmarks[i].y * h) for i in eyebrow_landmarks if i < len(landmarks)]
    eyebrow_y = min(eyebrow_y_values) if eyebrow_y_values else min(y_coords)
    
    # Define forehead boundaries - be more aggressive with bottom extension
    y_min = max(forehead_top - padding - 50, 0)  # Top of forehead - extended more aggressively
    y_max = max(eyebrow_y + 80, y_min + 100)  # Extend much further down, larger minimum height
    x_min = max(forehead_left - padding - 40, 0)  # Left side - extended even more
    x_max = min(forehead_right + padding + 100, w)  # Right side - extended even more
    
    # Ensure we have a valid crop
    if y_max <= y_min:
        y_max = y_min + 150  # Larger minimum height
    
    return image[y_min:y_max, x_min:x_max]


if __name__ == "__main__":
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            face_crop = crop_part(image, landmarks, FACE)
            left_eye_crop = crop_part(image, landmarks, LEFT_EYE, padding=27)
            right_eye_crop = crop_part(image, landmarks, RIGHT_EYE, padding=27)
            forehead_crop = crop_forehead(image, landmarks, FOREHEAD, padding=30)

            cv2.imwrite("face.png", face_crop)
            cv2.imwrite("left_eye.png", left_eye_crop)
            cv2.imwrite("right_eye.png", right_eye_crop)
            cv2.imwrite("forehead.png", forehead_crop)
