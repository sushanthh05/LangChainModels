from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv      
load_dotenv()
chat_model = ChatAnthropic(model="claude-3.5-sonnet",temperature=0.9,max_completion_tokens=100)

res=chat_model.invoke("What is the capital of India?")  
print(res.content)  