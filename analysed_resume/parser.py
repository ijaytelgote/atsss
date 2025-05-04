import json
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()
groq_api_key = os.getenv('API_KEY')

# Define your desired data structure.
from pydantic import BaseModel, Field
from typing import Optional

class Parser(BaseModel):
    experience: Optional[str] = Field(
        None,
        description="Extract the Experience section from the resume, including dates and designations if available. If the Experience section is not present, return `None`."
    )
    projects: Optional[str] = Field(
        None,
        description="Extract the Projects section from the resume. If the Projects section is not present, return `None`."
    )
    skills: Optional[str] = Field(
        None,
        description="Extract the Skills (technical skills) section from the resume. If the Skills section is not present, return `None`."
    )

# And a query intented to prompt a language model to populate the data structure.
def parser(resume):
    
    parser = JsonOutputParser(pydantic_object=Parser)

    prompt = PromptTemplate(
        template='''output must be the factually present in resume, not a dummy one.\n{format_instructions}\n{query}\n''',
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    model = ChatGroq(temperature=0, groq_api_key=groq_api_key, 
                    model_name="Gemma2-9b-It")

    chain = prompt | model | parser

    output=chain.invoke({"query": resume})
    return  output
from typing import Optional

class ResumeParser(BaseModel):
    experience: Optional[str] = Field(
        None,
        description="Extract the Experience section (including dates and designations) only if it is present in the resume; otherwise, return `None`."
    )
    projects: Optional[str] = Field(
        None,
        description="Extract the Projects section only if it is present in the resume; otherwise, return `None`."
    )
    skills: Optional[str] = Field(
        None,
        description="Extract the Skills (technical skills) section only if it is present in the resume; otherwise, return `None`."
    )
    education: Optional[str] = Field(
        None,
        description="Extract the Education section (including dates and degrees) only if it is present in the resume; otherwise, return `None`."
    )
    summary: Optional[str] = Field(
        None,
        description="Extract the Summary section only if it is present in the resume; otherwise, return `None`."
    )
    certifications: Optional[str] = Field(
        None,
        description="Extract the Certifications section (including dates and designations) only if it is present in the resume; otherwise, return `None`."
    )
    awards: Optional[str] = Field(
        None,
        description="Extract the Awards section (including dates and designations) only if it is present in the resume; otherwise, return `None`."
    )
    languages: Optional[str] = Field(
        None,
        description="Extract the Languages section only if it is present in the resume; otherwise, return `None`."
    )
    hobbies: Optional[str] = Field(
        None,
        description="Extract the Hobbies section only if it is present in the resume; otherwise, return `None`."
    )
    references: Optional[str] = Field(
        None,
        description="Extract the References section only if it is present in the resume; otherwise, return `None`."
    )
    publications: Optional[str] = Field(
        None,
        description="Extract the Publications section (including dates and designations) only if it is present in the resume; otherwise, return `None`."
    )
    additional_information: Optional[str] = Field(
        None,
        description="Extract the Additional Information section only if it is present in the resume; otherwise, return `None`."
    )
    personal_information: Optional[dict] = Field(
        None,
        description="Extract the Personal Information section (e.g., name of person, email, phone, address) only if it is present in the resume; otherwise, return `None`."
    )
    social: Optional[dict] = Field(
        None,
        description="Extract the Social Media Links section (e.g., LinkedIn, GitHub) only if it is present in the resume; otherwise, return `None`."
    )
def parse_whole_resume(resume):
    
    parser = JsonOutputParser(pydantic_object=ResumeParser)

    prompt = PromptTemplate(
        template='''output must be the factually present in resume, not a dummy one.\n{format_instructions}\n{query}\n''',
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    model = ChatGroq(temperature=0, groq_api_key=groq_api_key,
                    model_name="llama3-70b-8192")

    chain = prompt | model | parser

    output=chain.invoke({"query": resume})
    return  output
