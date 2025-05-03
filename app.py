import base64
import logging
import os
import sys
import threading

import bcrypt
import fitz  # PyMuPDF
import gdown
from main import list_down_reasons

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysed_resume.main import root
from cover_letter import cv_gen
from flask import Flask, jsonify, request, send_file
from main import (all_other, education_master, finale, last_score,
                  logic_actionable_words, logic_similarity_matching2,
                  main_score, master_score, resume_parsing_2, to_check_exp)

logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

country = 'India'
class User:
    username=None
from opt import *

def ensure_all_scores(score_dict, required_keys):
    for key in required_keys:
        if key not in score_dict:
            score_dict[key] = 0

def extract_text_from_pdf(pdf_file):
    try:
        pdf_content = pdf_file.read()
        pdf_file.seek(0)
        pdf_text = ""
        with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
            for page in pdf:
                pdf_text += page.get_text()
        return pdf_text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return None

def run_thread(target, *args):
    try:
        thread = threading.Thread(target=target, args=args)
        thread.start()
        thread.join()
    except Exception as e:
        logging.error(f"Error in {target.__name__}: {e}")

def getter(pdf_path):
    pdf_path.seek(0)
    pdf_data = pdf_path.read()
    pdf_path.seek(0)
    return pdf_path.filename,  pdf_data

def getter2(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_data=file.read()
    return pdf_data
def start(pdf_path, text_input,exp):
    resume = extract_text_from_pdf(pdf_path)
    jd = text_input

    if resume and jd:
        data45=getter(pdf_path)
        metadata = {
            'username': User.username,
            'pdf_name': str(data45[0]),
            'pdfBase64': base64.b64encode(data45[1]).decode('utf-8'),
            }
        store_pdf(metadata)
        run_thread(education_master, resume, master_score, country)
        run_thread(finale, resume, master_score)
        run_thread(resume_parsing_2, resume, master_score,exp)
        run_thread(to_check_exp, resume, jd, main_score,exp)
        run_thread(logic_actionable_words, resume, master_score)
        run_thread(logic_similarity_matching2, resume, jd, master_score)
        run_thread(all_other, master_score, pdf_path)

        required_master_keys = [
            'score_education_detection_', 'score_other',
            'similarity_matching_score', 'Action_score',
            'matrix_score'
        ]
        required_main_keys = ['exp_match', 'Parsing_score']

        ensure_all_scores(master_score, required_master_keys)
        ensure_all_scores(main_score, required_main_keys)

        all_score = [
            master_score['score_education_detection_'],
            master_score['score_other'],
            master_score['similarity_matching_score'],
            master_score['Action_score'],
            master_score['matrix_score']
        ]
        if_resume = master_score['Parsing_score']

        work_exp_matches = main_score['exp_match']
        scoring = last_score(all_score, work_exp_matches)
        if if_resume == 1:
            logging.info("Resume parsed successfully")
            if isinstance(scoring, str):
                current_value = float(scoring.strip('%'))
            else:
                current_value = float(scoring)
        else:
            logging.info(f"Resume not parsed: {if_resume}")
            current_value = 0
        master_score.update({'exp_match': main_score['exp_match']})
        li={}
        li['ss']=current_value
        li['li']=list(set(list_down_reasons()))

        return li





def parse(link):
    codes=link.split('/')
    for i in codes:
        if len(i)==33:
            return i
    return None



from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/coverLetter', methods=['POST'])
def coverLtter():
    pdf_file = request.files.get('pdf_file')
    jd = request.form.get('jd')  
    if not pdf_file or not jd:
        return jsonify({"error": "PDF file and Job Description are required"}), 400

    coverL=cv_gen(extract_text_from_pdf(pdf_file),jd)
    if coverL:
        output=jsonify({'cover_letter':coverL})
        return output
    
    
@app.route('/calculate_score', methods=['POST'])
def calculate_score():
    pdf_file = request.files.get('pdf_file')
    jd = request.form.get('jd')  
    if not pdf_file or not jd:
        return jsonify({"error": "PDF file and Job Description are required"}), 400

    metadata=root(extract_text_from_pdf(pdf_file),jd)
    if metadata== 'Invalid_JD':
        return jsonify({"error": "Invalid JD"}), 400

    if not metadata or not metadata.get('parsed_exp'):
        return jsonify({"error": "Not a valid resume"}), 400    
    score = start(pdf_file, jd, metadata['parsed_exp'])
    if metadata:
        output=jsonify({'score':score['ss'],'metadata':metadata,'resume_specific':score['li']})
    else:
        output=jsonify({'score':score['ss'],'resume_specific':score['li']})
    return output


if __name__ == '__main__':
    # Bind to 0.0.0.0 to allow external connections (necessary for Render)
    app.run(debug=False, threaded=True, use_reloader=False)

# #waitress-serve --listen=0.0.0.0:5000 flask_app:app
