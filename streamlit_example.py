import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import webbrowser
import random

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
    st.write('This is the home page. Check out our latest updates and activities here.')

    st.subheader('Interesting Machine Learning Fact/Tip of the Day')
    display_random_fact()

elif page == 'Events':
    st.header('Events')
    st.write('Explore our upcoming and past events related to machine learning.')

    # Define event details with images
    events = [
        {
            'title': 'Workshop on Neural Networks',
            'date': 'January 15, 2023',
            'image': 'image.jpg',
            'description': 'Join us for an in-depth exploration of neural networks.'
        },
        {
            'title': 'Guest Lecture on AI Ethics',
            'date': 'February 5, 2023',
            'image': 'image.jpg',
            'description': 'A thought-provoking discussion on the ethical implications of AI.'
        }
        # Add more event details with images as needed
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

elif page == 'About Us':
    st.header('About Us')
    st.write('Learn more about our club and its mission.')

    st.subheader('Lead')
    lead_info = {
        'name': 'Virat Kohli',
        'position': 'President',
        'bio': 'An experienced machine learning enthusiast leading the club towards innovation.',
        'image': 'image.jpg'  # Replace with the lead's image filename
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
        {'name': 'Virat Kohli', 'position': 'Software Engineer', 'image': 'image.jpg'},
        {'name': 'Virat Kohli', 'position': 'Data Scientist', 'image': 'image.jpg'},
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
        {'name': 'Virat Kohli', 'position': 'Public Relations Manager', 'image': 'image.jpg'},
        {'name': 'Virat Kohli', 'position': 'Social Media Coordinator', 'image': 'image.jpg'},
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
