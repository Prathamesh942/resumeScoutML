import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for uploading (only PDFs)
ALLOWED_EXTENSIONS = {'pdf'}

# Load the saved Random Forest model
model = joblib.load('Random_Forest.pkl')

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

# Function to calculate cosine similarity
def calculate_similarity(resume_text, job_desc_text):
    vectorizer = model.named_steps['preprocessor'].named_transformers_['text'].named_steps['tfidf']
    resume_vec = vectorizer.transform([resume_text])
    job_desc_vec = vectorizer.transform([job_desc_text])
    return cosine_similarity(resume_vec, job_desc_vec)[0][0]

# Function to predict fit status
def get_fit_status(resume_text, job_desc_text):
    similarity_score = calculate_similarity(resume_text, job_desc_text)
    feature_vector = {
        'resume':  resume_text + " " + job_desc_text,
        'similarity_score': similarity_score
    }
    prediction = model.predict(pd.DataFrame([feature_vector]))
    return 'Fit' if prediction[0] == 2 else ('Medium Fit' if prediction[0] == 1 else 'Not Fit')

# Route for uploading resumes and job description
@app.route('/', methods=['GET', 'POST'])
def upload_resumes():
    if request.method == 'POST':
        job_description = request.form['jobDescription']
        resume_files = request.files.getlist("resume_folder")

        if not resume_files or not job_description:
            return 'No resumes or job description provided.'

        results = []
        for file in resume_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                resume_text = extract_text_from_pdf(filepath)
                fit_status = get_fit_status(resume_text, job_description)
                similarity_score = calculate_similarity(resume_text, job_description)
                results.append((filename, fit_status, similarity_score))

        return render_template('results.html', results=results)

    return render_template('upload.html')

# Run the app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
