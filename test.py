import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate

# load_dotenv()
# apikey = os.environ["OPENAI_API_KEY"] 
llm = OpenAI(openai_api_key="")
query = llm.invoke("Hello")
print(query)