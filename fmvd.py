import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0.5)

# Function to draw points on the face
def draw_face_points(image, face_landmarks):
    for i, landmark in enumerate(face_landmarks.landmark):
        h, w, _ = image.shape
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)
        print(f"Point {i + 1}: ({cx}, {cy})")

# For video input:

cap = cv2.VideoCapture("vd1.mp4")

# Set desired width and height for the displayed frame
display_width = 540
display_height = 980

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Reached end of video.")
            break

        # Resize the image to the desired display dimensions
        image = cv2.resize(image, (display_width, display_height))

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                draw_face_points(image, face_landmarks)

        # Display the frame with face landmarks.
        cv2.imshow('MediaPipe Face Mesh', image)

        # Press 'q' to exit the video playback.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
