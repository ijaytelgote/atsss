import requests

# Replace with the port you are exposing in Docker
API_URL = "http://localhost:5001/calculate_score"

# Path to a sample resume PDF file on your local machine
PDF_FILE_PATH = r"C:\Users\jayma\Downloads\Shubham_murar_resume (3).pdf"

# Sample Job Description text
JOB_DESCRIPTION = """
We are looking for a Software Engineer with experience in Python, Machine Learning, and REST API development.
"""

def test_calculate_score():
    # Prepare files and data
    files = {'pdf_file': open(PDF_FILE_PATH, 'rb')}
    data = {'jd': JOB_DESCRIPTION}

    # Send POST request
    response = requests.post(API_URL, files=files, data=data)

    # Output response
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(response.json())
    except Exception as e:
        print("Response Text:")
        print(response.text)

if __name__ == "__main__":
    test_calculate_score()
