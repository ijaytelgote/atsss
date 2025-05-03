import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('API_KEY')

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='llama3-70b-8192'
)

def cv_gen(job_description: str, resume: str) -> str:
    """
    Generates a professional cover letter based on the provided job description and resume.

    Args:
        job_description (str): The job description for the position.
        resume (str): The applicant's resume.

    Returns:
        str: The formatted cover letter as a string with proper new lines (\n).
    """
    # Define the system prompt
    system_prompt = '''
You are an expert Cover Letter Creator with over 5 years of experience in crafting and optimizing resumes and cover letters. 
Your task is to create a professional, tailored cover letter based on the job description and resume provided. Follow the instructions below to ensure the cover letter meets industry standards:

Structure:
- Professional Introduction:
  - Clearly mention the position being applied for and where the job was found.
  - Briefly express enthusiasm for the role and organization.
- First Body Paragraph:
  - Highlight key qualifications, hard skills, or accomplishments relevant to the role.
  - Use specific examples to demonstrate experience and achievements.
- Second Body Paragraph:
  - Emphasize soft skills like communication, leadership, or teamwork.
  - Explain how these align with the company’s culture and the job’s requirements.
- Closing Paragraph:
  - Express availability for an interview or further discussion.
  - Include contact information and show gratitude for the hiring manager's time.

Tone:
- Use a confident, positive, and professional tone.
- Incorporate industry-specific language and action-oriented verbs.
- Ensure the letter is error-free and properly formatted.

Formatting:
- Add proper new line characters (\n) for every line break.
- Format the letter neatly and professionally.

Output:
- Provide the finalized cover letter formatted with proper new lines (\n) without any additional explanation or preamble
- start your answer with ``Here is your cover letter:``.
'''

    # Define the human prompt
    human_prompt = '''
This is the job description: \n{job_description}\n
This is the resume: \n{resume}\n
Use this date on the cover letter: \n{current_date}
'''

    # Prepare the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    # Combine the prompts and LLM into a chain
    chain = prompt | llm

    # Invoke the chain to generate the cover letter
    current_date = datetime.now().strftime("%d %B %Y")
    response = chain.invoke({
        "job_description": job_description,
        "resume": resume,
        "current_date": current_date
    })
    try:
      return response.content.split("Here is your cover letter:")[1]
    except Exception as e:
      print(f"Error: {e}")
      return response.content
 

# Example usage:
# jd = "Your job description text here."
# resume = "Your resume text here."
# print(generate_cover_letter(jd, resume))
