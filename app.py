import base64
import logging
import os
import sys
import threading
import bcrypt
import fitz  # PyMuPDF
import gdown
from main import list_down_reasons
from analysed_resume.main import root
from cover_letter import cv_gen
from flask import Flask, jsonify, request, send_file
from main import (all_other, education_master, finale, last_score,
                  logic_actionable_words, logic_similarity_matching2,
                  main_score, master_score, resume_parsing_2, to_check_exp)
import firebase_admin
from firebase_admin import credentials, db



logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
country = 'India'



creadd={
'type' : os.getenv(type)
'project_id' : os.getenv(project_id)
'private_key_id' : os.getenv(private_key_id)
'private_key' : os.getenv(private_key)
'client_email' : os.getenv(client_email)
'client_id' : os.getenv(client_id)
'auth_uri' : os.getenv(auth_uri)
'token_uri' : os.getenv(token_uri)
'auth_provider_x509_cert_url' : os.getenv(auth_provider_x509_cert_url)
'client_x509_cert_url' : os.getenv(client_x509_cert_url)
'universe_domain' : os.getenv(universe_domain)
}


if not firebase_admin._apps:
    cred = credentials.Certificate(creadd)
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://atscore-f8c1a-default-rtdb.firebaseio.com/'})
ref = db.reference('AUTH')
ref2 = db.reference('HISTORY')
ref3 = db.reference('REPORT')
ref4 = db.reference('SCAM_UPDATES')



class User:
    username=None
from opt import *


def hashed(username,password):
    mashed=str(username)+str(password)
    bytes = mashed.encode('utf-8')   
    salt = bcrypt.gensalt() 
    hash = bcrypt.hashpw(bytes, salt) 
    return hash.decode('utf-8')

def check_hashed(hash,new_password):
    decoded_hash=hash.encode('utf-8')
    return bcrypt.checkpw(new_password, decoded_hash)






def signup_(metadata):
    item_ref = ref.child(metadata.get('username'))  # Use the custom key
    item_data = item_ref.get()  # Get current data for the item

    if item_data:
        return "User already exists"
    new_item_data = {
        "nick_id": metadata['nick'],
        'username':metadata['username'],
        'email':metadata.get('email')
    }
    item_ref.set(new_item_data)

def get_email():
    username=User.username
    item_ref = ref.child(username)  
    item_data = item_ref.get()  
    print(item_data)
    if item_data:
        email=item_data.get('email')
        if email:
            return email
        return None
    return 'Please login does not exists'


    
def signin_(metadata):
    User.username=str(metadata.get('username'))
    item_ref = ref.child(metadata.get('username'))  
    item_data = item_ref.get()  
    if item_data:
        if check_hashed(item_data['nick_id'],metadata.get('nick')) :
            return True
        return None
    return 'User does not exists'

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



def store_pdf(metadata):
    data = ref2.child(metadata.get('username'))
    item = data.get()
    item ={"pdf_name":metadata.get('pdf_name'),"pdfBase64":metadata.get('pdfBase64')}
    data.set(item)
    logging.info("PDF has been saved")
    pass

def get_stored_pdf(metadata):
    data = ref2.child(metadata.get('username'))
    item = data.get()
    return item



def get_saved_pdf():
    metadata = {'username': User.username}  #Adjust metadata as needed
    output = get_stored_pdf(metadata)
    pdfbase64=output.get('pdfBase64')
    file_name=output.get('pdf_name')
    return pdfbase64,file_name


def parse(link):
    codes=link.split('/')
    for i in codes:
        if len(i)==33:
            return i
    return None


def report_area(metadata):
    data = ref3.child(metadata.get('username'))
    item = data.get()
    if item:
        item.append(metadata.get('problem'))
        data.set(item)
        logging.info("new report has been saved")
    else:
        item =metadata.get('problem')
        data.set([item])
        logging.info("report has been saved")

    
def scam_area(metadata):
    data = ref4.child(metadata.get('username'))
    item = data.get()
    if item:
        item.append(metadata.get('scam'))
        data.set(item)
        logging.info("new scam has been saved")
    else:
        item =metadata.get('scam')
        data.set([item])
        logging.info("scam has been saved")

from flask_cors import CORS

app = Flask(__name__)

CORS(app)
@app.route("/report",methods=["POST"])
def sub_report():
    data=request.get_json()
    problem=data.get('problem')
    send_feedback_mail(receiver_email=get_email())  


    if not problem:
        return jsonify({"error": "Problem description is required."}), 400
    report_area({"username":User.username,"problem":problem})
    return jsonify({"message":"report has been saved"}), 200

@app.route("/submit_feedback",methods=["POST"])
def sub_scam():
    data=request.get_json()
    scam=data.get('feedback')
    if not scam:
        return jsonify({"status": "error", "message": "Feedback cannot be empty"})
    scam_area({"username":User.username,"scam":scam})
    return jsonify({"status": "success"})


@app.route('/gdrive_store', methods=['GET', 'POST'])
def gdrive_store():
    if request.method == 'POST':
        link = request.form.get('file_id')
        if not link:
            return jsonify({"error": "Error: link is required!"}), 400
        try:
            file_id = parse(link)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_file = os.path.join(UPLOAD_FOLDER, f'resume.pdf')   # Unique file name
        try:
            gdown.download(download_url, output_file, quiet=False)
            outer = getter2(output_file)
            metadata = {
                'username': User.username,
                'pdf_name': 'Resume.pdf',
                'pdfBase64': base64.b64encode(outer).decode('utf-8'),
            }
            store_pdf(metadata)
            ioss = get_saved_pdf()
            
            return jsonify({
                "pdfBase64": ioss[0],
                "file_name": ioss[1]
            }), 200
        except Exception as e:
            app.logger.error(f"Error processing file: {e}")
            return jsonify({"error": f"Failed to download file: {str(e)}"}), 500


@app.route('/saved_pdf', methods=['GET'])
def saved_pdf():
    try:
        ioss=get_saved_pdf()
        return jsonify({"pdfBase64":ioss[0],"file_name":ioss[1]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")
        email=data.get('email')
        ids_=hashed(username,password)
        metadata={
            'username':username,
            'email':email,
            'password':password,
            'nick':ids_,
            'intent':'sign_up'
            }
        o=signup_(metadata)
        if o=='User already exists':
            return jsonify({"message": "User already exists, please login"}), 200
        return jsonify({"message": "User created successfully", "uid": metadata['nick']}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/signin', methods=['POST'])
def signin():
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")
        ids_=(str(username)+str(password)).encode('utf-8')  
        metadata={
        'username':username,
        'password':password, 
        'nick':ids_,
        'intent':'sign_in'}
        o=signin_(metadata)
        if o:
            if o=='User does not exists':
                return jsonify({"message": "User does not exists, please signup"}), 200
            else:
                return jsonify({"message": "Signed in Successfully","token":str(ids_)}), 200
        else:
            return jsonify({"message": "Signed in Unsuccessfully"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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
