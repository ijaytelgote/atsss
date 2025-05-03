import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()
groq_api_key = os.getenv('API_KEY')

# Define your desired data structure.
class XYZ(BaseModel):
    before: str = Field(description="Before (output in list)")
    after: str = Field(description="After using XYZ Formula (output in list)")

# And a query intented to prompt a language model to populate the data structure.
def xyz_resume(resume):
    
    parser = JsonOutputParser(pydantic_object=XYZ)

    prompt = PromptTemplate(
        template="""Rewrite the following responsibilities using Google's XYZ formula. 
            For each responsibility, quantify the accomplishment (X), the impact or result (Y), and the action taken (Z). 
            Ensure that each bullet point clearly communicates the contribution and outcome.

            Example Before:
                ["Led a team to develop a customer relationship management app.",
                "Improved data processing pipeline."]
            Example After using XYZ Formula:
                ["Led a team of 5 developers to build a customer relationship management app, increasing user retention by 25% within 3 months by implementing a new recommendation engine.",
                "Improved data processing pipeline, reducing processing time by 40% by optimizing the ETL workflows..\n"]
        {format_instructions}\n{query}\n
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    model = ChatGroq(temperature=0, groq_api_key=groq_api_key, 
                    model_name="Gemma2-9b-It")

    chain = prompt | model | parser

    output=chain.invoke({"query": resume})
    return  output
