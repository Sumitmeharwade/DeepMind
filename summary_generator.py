import streamlit as st
from urllib.parse import parse_qs, urlparse
import google.generativeai as palm
from apiclient.discovery import build
import pandas as pd
import random
from rouge_score import rouge_scorer

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
        request = f"from the given comments summarize what the people are talking about. Also, give the general tone of the comments. Give the output within 500 words: {self.text_content}"
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

def main():
    st.title("YouTube Comment Summarizer")

    video_url = st.text_input("Enter YouTube Video URL:")

    if st.button("Generate Summary"):
        summary = scrape_all_with_replies(video_url)
        st.success(summary)

if __name__ == "__main__":
    main()