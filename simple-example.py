from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_key=api_key)
prompt = "Translate the following English text to French: 'Je suis un homme.'"
response = llm.invoke(prompt)
print(response)
