from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(model_id="microsoft/Phi-3-mini-4k-instruct",
                                      task="text-generation",
                                      pipeline_kwargs={"temperature":0.5,"max_new_tokens":100})

model=ChatHuggingFace(llm=llm,temperature=0.9)
res=model.invoke("What is the capital of India?")

print(res.content)