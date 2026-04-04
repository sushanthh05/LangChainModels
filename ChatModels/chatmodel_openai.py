from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOpenAI(model="gpt-4",temperature=0.9,max_completion_tokens=100)

res=chat_model.invoke("What is the capital of India?")

print(res.content)