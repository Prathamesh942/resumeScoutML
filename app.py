import os
import re
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from joblib import load
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("punkt")
nltk.download("stopwords")

# Create Flask app
app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for uploading (only PDFs)
ALLOWED_EXTENSIONS = {'pdf'}

# Function to check if the file is a valid PDF
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Function to clean the resume text
def clean_resume(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

# Function to calculate cosine similarity
def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    combined_text = [resume_text, job_desc_text]
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score

# Function to determine fit status based on similarity score
def determine_fit_status(similarity_score):
    if similarity_score > 0.55:
        return 'Fit'
    elif similarity_score >= 0.35:
        return 'Medium Fit'
    else:
        return 'Not Fit'

# Function to predict resume category
def predict_resume(filepath):
    try:
        text = extract_text_from_pdf(filepath)
        text = clean_resume(text)
        text = [text]
        text = np.array(text)
        vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
        resume = vectorizer.transform(text)
        model = load('model.joblib')
        label_dict = {
            0: 'Arts', 1: 'Automation Testing', 2: 'Operations Manager',
            3: 'DotNet Developer', 4: 'Civil Engineer', 5: 'Data Science',
            6: 'Database', 7: 'DevOps Engineer', 8: 'Business Analyst',
            9: 'Health and fitness', 10: 'HR', 11: 'Electrical Engineering',
            12: 'Java Developer', 13: 'Mechanical Engineer', 14: 'Network Security Engineer',
            15: 'Blockchain', 16: 'Python Developer', 17: 'Sales',
            18: 'Testing', 19: 'Web Designing'
        }
        return label_dict[model.predict(resume)[0]]
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return "Unknown"

# Route for uploading resume and job description
@app.route('/', methods=['POST'])
def upload_resume():
    if request.method == 'POST':
        job_description = request.form.get('jobDescription')
        resume_id = request.form.get('resumeId')
        file = request.files.get("resume")

        if not file or not job_description or not resume_id:
            return jsonify({"error": "Missing resume, job description, or resume ID."}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            resume_text = extract_text_from_pdf(filepath)
            similarity_score = calculate_similarity(resume_text, job_description)
            fit_status = determine_fit_status(similarity_score)
            # predicted_category = predict_resume(filepath)

            return jsonify({
                "resumeId": resume_id,
                "fitStatus": fit_status,
                "similarityScore": similarity_score
            })

# Run the app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)