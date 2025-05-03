from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key = os.getenv('API_KEY')
model = ChatGroq(temperature=0, groq_api_key=groq_api_key, 
                    model_name="Gemma2-9b-It")
# Define your desired data structure for error detection in resumes
class GrammarCorrection(BaseModel):
    spelling_errors: list[str] = Field(description="List of detected spelling and grammer errors, if no errors are detected return `None`")
    corrected_text: list[str] = Field(description="List of spelling and grammer error correction., if no correction needed return `None`)")

def grammer(resume_text):
    parser = JsonOutputParser(pydantic_object=GrammarCorrection)

    prompt = PromptTemplate(
        template="""
        Review the following resume text for grammar and spelling mistakes and provide corrections
        Detect and correct grammar and spelling errors in the resume text provided.
        Dont consider Company name for correction, may be it would as like as.
        {format_instructions}
        Resume Text: {resume_text}
        """,
        input_variables=["resume_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    result = chain.invoke({"resume_text": resume_text})
    return result