import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import webbrowser
import random
import mediapipe as mp
from urllib.parse import parse_qs, urlparse
import google.generativeai as palm
from googleapiclient.discovery import build
import pandas as pd
import random
from rouge_score import rouge_scorer
import numpy as np



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
                stframe.image(cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB), channels="RGB")

            key = cv2.waitKey(1)
            if key == 27:  # 'esc' key
                video.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('f'):  # 'f' key
                is_full_screen = True
            elif key == ord('g'):  # 'g' key
                is_full_screen = False

    def run():
        stframe = st.empty()
        face_mesh = load_face_mesh()
        process_video(face_mesh, stframe)
    run()



# Set your YouTube API key and Google API key
YOUTUBE_API_KEY = "AIzaSyDVKROa2PHT7JrXg_bMqZ1-7HNCnxpqsZ8"
GOOGLE_API_KEY = "AIzaSyDxp5B2tqKHGOWjI0rp8pL1eyzJMiIdVac"

# Initialize the YouTube API
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

class PalmAI(object):
    def __init__(self, text_content):
        self.text_content = text_content
        self.generated_summary = ""

    def GenerateSummary(self):
        palm.configure(api_key=GOOGLE_API_KEY)
        request = f"from the given comments summarize what the people are talking about. Also, give the general tone of the comments. Give the output within 1000 words: {self.text_content}"
        response = palm.generate_text(prompt=request)
        self.generated_summary = response.result

def get_video_description(video_id):
    video_data = youtube.videos().list(part='snippet', id=video_id).execute()
    if 'items' in video_data and video_data['items']:
        return video_data['items'][0]['snippet']['description']
    else:
        return ''

def get_video_id_from_url(video_url):
    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get('v')
    return video_id[0] if video_id else None

def calculate_rouge(hypothesis, reference):
    if hypothesis is None or reference is None:
        return {'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(hypothesis, reference)
    return scores

def scrape_all_with_replies(video_url):
    video_id = get_video_id_from_url(video_url)

    if not video_id:
        return "Invalid YouTube Video URL. Please provide a valid URL."

    video_description = get_video_description(video_id)

    data = youtube.commentThreads().list(part='snippet', videoId=video_id, maxResults=100, textFormat="plainText").execute()
    print("Data:", data)

    comments_to_concatenate = []

    for i in data["items"]:
        name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
        comment = i["snippet"]['topLevelComment']["snippet"]["textDisplay"]
        likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
        published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
        replies = i["snippet"]['totalReplyCount']

        comments_to_concatenate.append(comment)

        TotalReplyCount = i["snippet"]['totalReplyCount']

        if TotalReplyCount > 0:
            parent = i["snippet"]['topLevelComment']["id"]

            data2 = youtube.comments().list(part='snippet', maxResults=100, parentId=parent, textFormat="plainText").execute()

            for i in data2["items"]:
                comment = i["snippet"]["textDisplay"]
                comments_to_concatenate.append(comment)

    selected_comments = random.sample(comments_to_concatenate, min(20, len(comments_to_concatenate)))
    concatenated_comments = ' '.join(selected_comments)

    summaryGenerator = PalmAI(concatenated_comments)
    summaryGenerator.GenerateSummary()

    generated_summary = summaryGenerator.generated_summary
    if generated_summary is not None:
        print("Generated Summary:", generated_summary)
    else:
        print("Error generating summary. Check for issues in the summary generation process.")

    reference_summary = "The people are discussing about Tailwind CSS. Some of them think that it is not necessary to learn Tailwind CSS and people can directly use it by reading the documentation. Others disagree and think that it is better to learn Tailwind CSS first. The overall tone of the comments is positive and helpful."
    rouge_scores = calculate_rouge(generated_summary, reference_summary)
    print("ROUGE Scores:", rouge_scores)

    if generated_summary is not None:
        return "Successful! Summary: " + generated_summary
    else:
        return "Error generating summary. Check for issues in the summary generation process."



def main_face_mesh_detection():
    st.title("Face Mesh Detection with OpenCV and MediaPipe")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    drawing = mp.solutions.drawing_utils
    spec = drawing.DrawingSpec(thickness=1, circle_radius=1)

    def detect_face_mesh():
        landmarks_list = []
        video = cv2.VideoCapture(0)
        stframe = st.empty()

        stop_flag = False
        stop_button = st.button("Stop Detection")

        while not stop_flag:
            _, frame = video.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            op = face_mesh.process(rgb)

            if op.multi_face_landmarks:
                for face_landmarks in op.multi_face_landmarks:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        height, width, _ = frame.shape
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        landmarks_list.append((x, y))

                    drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=spec,
                        connection_drawing_spec=spec,
                    )

            stframe.image(frame, channels="RGB", use_column_width=True)

            if stop_button:
                stop_flag = True

        video.release()
        cv2.destroyAllWindows()

    detect_face_mesh()

class ObjectDetector(VideoTransformerBase):
    def __init__(self):
        self.thres = 0.4  # Threshold to detect object
        self.classNames = []
        self.classFile = 'coco.names'
        with open(self.classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        self.configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weightsPath = 'frozen_inference_graph.pb'

        self.net = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.group_invite_opened = False  # Flag to track if the group invite link has been opened

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        classIds, confs, bbox = self.net.detect(img, confThreshold=self.thres)

        if not self.group_invite_opened:  # Check if invite link hasn't been opened
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if classId == 44:  # Checking for the classId of a bottle
                        webbrowser.open_new_tab("https://chat.whatsapp.com/BGtWD7jDZbXEo5BMqAjpOn")
                        self.group_invite_opened = True  # Set flag to True after opening link
                        break

                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=6)
                    cv2.putText(img, self.classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        return img

machine_learning_facts = [
    "Did you know? The term 'machine learning' was coined by Arthur Samuel in 1959.",
    "Tip: Feature scaling can improve the performance of many machine learning algorithms.",
    "Fun Fact: Neural networks were inspired by the structure of the human brain.",
    "Machine learning algorithms can be categorized into three types: supervised learning, unsupervised learning, and reinforcement learning.",
        "Regression and classification are two main types of supervised learning tasks.",
        "Anomaly detection, clustering, and dimensionality reduction are common unsupervised learning techniques.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn from data.",
        "Feature engineering involves selecting and transforming variables to improve the performance of machine learning models.",
        "Ensemble methods combine multiple machine learning models to enhance predictive performance.",
        "Bias-variance tradeoff is a key concept in machine learning that deals with model complexity and generalization.",
        "Overfitting occurs when a model learns too much from the training data and performs poorly on new, unseen data.",
        "Cross-validation is a technique used to assess a model's performance by splitting the data into multiple subsets for training and validation.",
        "A confusion matrix is a table used to evaluate the performance of a classification model.",
        "Precision and recall are important metrics in evaluating the performance of a classification model.",
        "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models.",
        "Transfer learning involves leveraging pre-trained models to perform tasks with limited data.",
        "Support Vector Machines (SVMs) are supervised learning models used for classification and regression tasks.",
        "Random Forest is an ensemble learning method that constructs multiple decision trees and merges their predictions.",
        "Natural Language Processing (NLP) focuses on enabling computers to understand, interpret, and generate human language.",
        "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment.",
        "K-means clustering is an unsupervised learning algorithm used for clustering data into distinct groups.",
        "Principal Component Analysis (PCA) is a dimensionality reduction technique used to reduce the number of features.",
        "Hyperparameters are parameters that are set before the learning process begins and affect the model's performance."
    # Add more facts or tips here
]

# Function to display a random fact or tip
def display_random_fact():
    random_fact = random.choice(machine_learning_facts)
    st.write(f"**Machine Learning Fact/Tip:** {random_fact}")


st.title('DeepMind!!!')
st.write('Welcome to the University Machine Learning Club! This app is a showcase of our activities.')

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ('Home', 'Events', 'Projects', 'About Us'))

if page == 'Home':
    st.header('Home')
    st.write('Check out our latest updates and activities here.')

    st.subheader('Interesting Machine Learning Fact/Tip of the Day')
    display_random_fact()

elif page == 'Events':
    st.header('Events')
    st.write('Explore our upcoming and past events related to machine learning.')

    # Define event details with images
    events = [
        {
            'title': 'Workshop on basics of ML and tools required',
            'date': 'Will be informed later',
            'image': 'image.jpg',
            'description': 'Join us for interactive sessions.'
        }
        # ,
        # {
        #     'title': 'Guest Lecture on AI Ethics',
        #     'date': 'February 5, 2023',
        #     'image': 'image.jpg',
        #     'description': 'A thought-provoking discussion on the ethical implications of AI.'
        # }
        # # Add more event details with images as needed
    ]

    # Display each event as a card
    for event in events:
        col1, col2 = st.columns([1, 4])
        with col1:
            event_image = f"images/{event['image']}"  # Replace with your image folder path
            st.image(event_image, use_column_width=True)
        with col2:
            st.subheader(event['title'])
            st.write(f"Date: {event['date']}")
            st.write(event['description'])

elif page == 'Projects':
    st.header('Projects - Object Detection')
    st.write('Show a bottle to join our WhatsApp group!')

    webrtc_ctx = webrtc_streamer(key="object-detection", video_transformer_factory=ObjectDetector)

    if webrtc_ctx.video_transformer:
        st.write('Object detection is active')

    # Thank you message after the invite link has been opened
    if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.group_invite_opened:
        st.write('Thank you for joining our WhatsApp group!')
        st.stop()  # Stop the app to prevent further processing
        st.header('Projects - Face Mesh Detection')
    st.write('Face Mesh Detection with OpenCV and MediaPipe')

    start_flag = st.button("Start Detection")

    if start_flag:
        main_face_mesh_detection()

    st.header('MediaPipe Face Mesh Demo')


    if st.checkbox('Start Face Mesh Detection'):
        face_mesh_project_page()

    st.header("YouTube Comment Summarizer")

    video_url = st.text_input("Enter YouTube Video URL:")

    if st.button("Generate Summary"):
        summary = scrape_all_with_replies(video_url)
        st.success(summary)
        
elif page == 'About Us':
    st.header('About Us')
    st.write('Learn more about our club and its mission.')

    st.subheader('Lead')
    lead_info = {
        'name': 'Ravishankar',
        'position': 'President',
        'bio': 'An experienced machine learning enthusiast leading the club towards innovation.',
        'image': 'ravvi.jpeg'  # Replace with the lead's image filename
    }
    col1, col2 = st.columns([1, 4])
    with col1:
        lead_image = f"images/{lead_info['image']}"  # Replace with your image folder path
        st.image(lead_image, use_column_width=True)
    with col2:
        st.subheader(lead_info['name'])
        st.write(f"Position: {lead_info['position']}")
        st.write(lead_info['bio'])

    st.subheader('Tech Team')
    tech_team_members = [
        {'name': 'Kushalgouda Patil', 'position': 'Tech lead', 'image': 'kush.jpeg'},
        {'name': 'Sumit Meharwade', 'position': '', 'image': 'sumit.jpg'},
        {'name': 'Neelkant', 'position': '', 'image': 'neel.jpg'},
        {'name': 'Sushant', 'position': '', 'image': 'sush.jpeg'},
        # Add more tech team members as needed
    ]
    col_count = 0
    for member in tech_team_members:
        col_count += 1
        col = st.columns(5) if col_count % 5 == 0 else st.columns(4)
        with col[col_count % 5 - 1]:
            member_image = f"images/{member['image']}"  # Replace with your image folder path
            st.image(member_image, use_column_width=True)
            st.subheader(member['name'])
            st.write(member['position'])

    st.subheader('PR/Media Team')
    pr_media_team_members = [
        {'name': 'Pratiksha Tigadi', 'position': 'Public Relations Manager', 'image': 'image.jpg'},
        {'name': 'Yashasvi', 'position': 'Coordinator', 'image': 'image.jpg'},
        # Add more PR/Media team members as needed
    ]
    col_count = 0
    for member in pr_media_team_members:
        col_count += 1
        col = st.columns(4)
        with col[col_count - 1]:
            member_image = f"images/{member['image']}"  # Replace with your image folder path
            st.image(member_image, use_column_width=True)
            st.subheader(member['name'])
            st.write(member['position'])

st.sidebar.markdown('---')
st.sidebar.write('© 2023 DeepMind Club')
