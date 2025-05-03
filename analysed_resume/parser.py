import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()
groq_api_key = os.getenv('API_KEY')

# Define your desired data structure.
class Parser(BaseModel):
    experience: str = Field(description="extract the Experience section from the resume if it contains or the mentioned(including dates and designation), it must be preset in resume if not return `None`.")
    projects: str = Field(description="extract the only Project section from the resume, it must be preset in resume if not return `None`.")
    skills: str = Field(description="extract the Skills (technical skills) section from the resume, it must be preset in resume if not return `None`.")

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
