import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

def home_page():
    st.title("Welcome to Face Mesh Detection Project!")
    st.write(
        "This is a simple Streamlit application that demonstrates face mesh detection using the MediaPipe library."
    )
    st.write("To view the Face Mesh Detection project, navigate to the 'Project Page'.")

def face_mesh_project_page():
    # Function to merge two frames side by side (i.e., input and output frames)
    def addFrame(img1, img2):
        r1, c1 = img1.shape[0:2]
        r2, c2 = img2.shape[0:2]

        result = np.zeros((max(r1, r2), c1 + c2, 3), dtype=np.uint8)

        result[:r1, :c1, :] = img1
        result[:r2, c1:c1 + c2, :] = img2

        return result

    # MediaPipe objects
    facemesh = mp.solutions.face_mesh
    draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Function to load FaceMesh model with caching
    @st.cache(allow_output_mutation=True)
    def load_face_mesh():
        return facemesh.FaceMesh(
            static_image_mode=True,
            min_tracking_confidence=0.6,
            min_detection_confidence=0.6,
            refine_landmarks=True
        )

    # Function to process the video stream
    def process_video(face, stframe):
        video = cv2.VideoCapture(0)
        is_full_screen = False

        while True:
            ret, frame = video.read()

            if not ret:
                st.write('Video processing complete.')
                break

            height, width, channels = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            op = face.process(rgb)

            if op.multi_face_landmarks:
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

                window_image = addFrame(frame, mesh_window)
                stframe.image(cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), channels="BGR")

            key = cv2.waitKey(1)
            if key == 27:  # 'esc' key
                video.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('f'):  # 'f' key
                is_full_screen = True
            elif key == ord('g'):  # 'g' key
                is_full_screen = False

    st.title('MediaPipe Face Mesh Demo')

    # Display the video stream in Streamlit
    stframe = st.empty()

    if st.checkbox('Start Face Mesh Detection'):
        face_mesh = load_face_mesh()
        process_video(face_mesh, stframe)

def main():
    st.set_page_config(
        page_title="Face Mesh Detection Project",
        page_icon=":sunglasses:",
        layout="wide"
    )

    pages = {
        "Home": home_page,
        "Face Mesh Detection Project": face_mesh_project_page
    }

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selected_page]()

if __name__ == "__main__":
    main()
