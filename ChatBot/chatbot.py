from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)

# Store conversation history
chat_history = []

# Create a prompt template that includes chat history
prompt = PromptTemplate(
    input_variables=["history", "user_input"],
    template="{history}User: {user_input}\nAssistant:"
)

chain = prompt | model | StrOutputParser()

while True:
    user_input = input('You: ')
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. Goodbye!")
        break
    
    # Build conversation history string
    history_str = ""
    for msg in chat_history:
        history_str += msg + "\n"
    
    response = chain.invoke({"history": history_str, "user_input": user_input}).strip()
    
    # Extract just the assistant's response
    if "Assistant:" in response:
        assistant_response = response.split("Assistant:")[-1].strip()
        if "\nUser:" in assistant_response:
            assistant_response = assistant_response.split("\nUser:")[0].strip()
    else:
        assistant_response = response
    
    # Add to chat history
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Assistant: {assistant_response}")
    
    print(f'ChatBot: {assistant_response}')
