import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Load the image
image_path = './data/face_01.jpg'
image = cv2.imread(image_path)

if image is None:
    print("❌ Failed to load image.")
    exit()

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image
results = face_mesh.process(image_rgb)

# Annotate bounding box around eyes
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape

        # Left eye landmarks (approximate region)
        left_eye_indices = [33, 133, 160, 159, 158, 144, 153, 154, 155]
        left_eye_points = [(int(face_landmarks.landmark[i].x * w),
                            int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]

        # Right eye landmarks (approximate region)
        right_eye_indices = [362, 263, 387, 386, 385, 373, 380, 374, 381]
        right_eye_points = [(int(face_landmarks.landmark[i].x * w),
                             int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]

        # Get bounding box for left eye
        lx, ly, lw, lh = cv2.boundingRect(np.array(left_eye_points))
        cv2.rectangle(image, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
        cv2.putText(image, 'Left Eye', (lx - 30, ly - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Get bounding box for right eye
        rx, ry, rw, rh = cv2.boundingRect(np.array(right_eye_points))
        cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        cv2.putText(image, 'Right Eye', (rx + 10, ry - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    # Save the annotated image
    cv2.imwrite('annotated_eyes_box.jpg', image)
    print("✅ Annotated image saved as 'annotated_eyes_box.jpg'")
else:
    print("❌ No face landmarks detected.")
