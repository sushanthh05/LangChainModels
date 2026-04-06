from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

emb=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

res=emb.embed_documents(["Delhi is the capital of India",
                         "Mumbai is the financial capital of India",
                         "Bangalore is the IT capital of India"])

print(str(res))
