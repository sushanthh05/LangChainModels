from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-pro",temperature=0.9,max_completion_tokens=100)

res=chat_model.invoke("What is the capital of India?")

print(res.content)