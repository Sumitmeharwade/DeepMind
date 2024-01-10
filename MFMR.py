import cv2
import mediapipe as mp
import numpy as np

# Function to merge two frames side by side (i.e., input and output frames)
def addFrame(img1, img2):
    r1, c1 = img1.shape[0:2]
    r2, c2 = img2.shape[0:2]

    result = np.zeros((r2, c1 + c2, 3), dtype=np.uint8)

    result[:, :c1, :] = img1
    result[:, c1:c1 + c2, :] = img2

    return result

# Video file
video = cv2.VideoCapture(0)

# MediaPipe objects
facemesh = mp.solutions.face_mesh
draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

landmark1 = []
landmarks = []
face = facemesh.FaceMesh(
    static_image_mode=True,
    min_tracking_confidence=0.6,
    min_detection_confidence=0.6,
    refine_landmarks=True
)

# Create an initial window for the video stream
cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_NORMAL)

# Process each frame
is_full_screen = False
while True:
    ret, frame = video.read()

    if not ret:
        print('Video processing complete.')
        break

    height, width, channels = frame.shape

    # MediaPipe requires RGB format
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Landmarks acquired in op
    op = face.process(rgb)

    # If no face detected
    if not op.multi_face_landmarks:
        continue

    # Extracting face landmarks
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            if (i.landmark[0] is not None and i.landmark[1] is not None):
                landmarks = []
                landmarks.append(i.landmark[0].y * 480)
                landmarks.append(i.landmark[1].x * 640)
                landmarks.append(i.landmark[2].z)
                landmark1.append(landmarks)

                # Corrected line
                mesh_window = np.zeros_like(frame)
                draw.draw_landmarks(
                    image=mesh_window,
                    landmark_list=op.multi_face_landmarks[0],
                    connections=facemesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                draw.draw_landmarks(
                    image=mesh_window,
                    landmark_list=op.multi_face_landmarks[0],
                    connections=facemesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                draw.draw_landmarks(
                    image=mesh_window,
                    landmark_list=op.multi_face_landmarks[0],
                    connections=facemesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

                # Merge the face mesh window with the original frame
                window_image = addFrame(mesh_window, frame)

                # Show the merged image in full screen or normal window mode
                if is_full_screen:
                    cv2.setWindowProperty('MediaPipe Face Mesh', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('MediaPipe Face Mesh', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(window_image, 1))

    # Check for key presses
    key = cv2.waitKey(1)
    if key == 27:  # 'esc' key
        video.release()
        cv2.destroyAllWindows()
        break
    elif key == ord('f'):  # 'f' key
        is_full_screen = True
    elif key == ord('g'):  # 'g' key
        is_full_screen = False
print('Completed successfully')
