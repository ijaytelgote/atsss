import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
# api="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"
groq_api_key = os.getenv('API_KEY')

client = Groq(api_key=groq_api_key)
class process:
    def skill_specific(self, resume, jd):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a Resume Analyzer. Your task is to analyze the resume provided and compare it with the job description (JD). Based on this, you will categorize the skills into 'Non-Relevant Skills' and suggest missing but relevant skills.

                    **Objective:**
                    1. Identify the skills that are irrelevant to the job description and list them under 'Non-Relevant Skills.'
                    2. Suggest relevant skills that may be missing from the resume but are important based on the job description.

                    **Instructions:**
                    - Review the resume: {resume}
                    - Review the job description: {jd}
                    - Compare the resume's skills with the JD and categorize them under 'Non-Relevant Skills' and 'Suggestions for Relevant Skills.'

                    **Response Format:**

                    **Non-Relevant Skills:**
                    - [Skill Name]: [Reason why the skill is not relevant to the job description]

                    **Suggestions for Relevant Skills:**
                    - [Skill Name]: [Why this skill would be relevant based on the job description]

                    **Important Guidelines:**
                    1. Only analyze technical skills (e.g., programming languages, databases, tools, frameworks).
                    2. If no non-relevant skills are found, state 'None.'
                    3. Avoid including any personal information or soft skills.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Please categorize the skills in the following resume:

                    **Resume:**
                    {resume}

                    **Job Description:**
                    {jd}

                    1. Identify and list the technical skills from the resume that are not relevant to the job description.
                    2. Provide suggestions for skills that may be missing from the resume but are essential for the job based on the JD.
                    3. follow the template strictly. dont give any information about from template
                    4. Directly return `Invalid_JD` without anything, if you find the given JD is not a valid Job Description.
                                        
                    """
                    }
            ],
            model="Gemma2-9b-It"
        )

        return chat_completion.choices[0].message.content





    def project_specific(self,resume, jd):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a Resume Analyzer. Analyze the provided resume against the job description (JD) and categorize the projects accordingly.
                    
                    **Objective:**
                    1. Determine which projects from the resume are relevant to the job description and list them under 'Relevant Projects.'
                    2. Identify any projects that are not relevant to the job description and list them under 'Non-Relevant Projects.'
                    3. Suggest alternative or additional projects that would align better with the job description if necessary.

                    **Instructions:**
                    - Review the resume: {resume}
                    - Review the job description: {jd}
                    - Categorize the projects clearly into 'Relevant' and 'Non-Relevant' sections based on the JD, role, and responsibilities.
                    
                    **Response Template:**

                    **Relevant Projects:**
                    - [Project Name]: [Reason why the project is relevant to the job description]

                    **Non-Relevant Projects:**
                    - [Project Name]: [Reason why the project is not relevant to the job description]

                    **Suggestions for Relevant Projects:**
                    - [Suggestion 1]: [Explanation of why this project would be better aligned with the JD]
                    - [Suggestion 2]: [Additional project recommendations based on the job requirements]

                    Example:
                    **Relevant Projects:**
                    - Customer Relationship Management App with Business Intelligence: This project aligns with the job's requirement for data analytics and machine learning skills.

                    **Non-Relevant Projects:**
                    - Resume Parsing and ATS Scoring software: This project is not aligned with the job's focus on adaptive control systems, which is a key requirement.

                    **Suggestions for Relevant Projects:**
                    - Consider a project involving training and validating deep learning models in real-world applications, which is a core focus of the job.
                    - Another suggestion could be projects demonstrating experience with classical and deep learning algorithms, as per the job requirements.

                    **Exclusions:**
                    1. Do not include any personal information or details irrelevant to the job description.
                    2. Avoid factually incorrect or extra information outside the context of the analysis.
                    3. If no projects are relevant or non-relevant, mention 'None' under the respective section.
                    """
                },
                {
                    "role": "user",
                    "content": """
                    1. It is critical to categorize projects into 'Relevant' and 'Non-Relevant' sections as per the JD.
                    2. Ensure projects are categorized after a detailed comparison with the job description, role, and responsibilities.
                    3. Provide your analysis in clear and concise points without irrelevant details.
                    4. If there are no relevant or non-relevant projects, only mention 'None' in the respective section withount any information or explaination.
                    5. Exclude unnecessary information like personal details or any content that violates the provided structure.
                    4. Directly return `Invalid_JD` without anything, if you find the given JD is not a valid Job Description.

                    """
                }
            ],
            model="Gemma2-9b-It",
        )

        return chat_completion.choices[0].message.content





