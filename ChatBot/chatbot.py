from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)

# System message to define chatbot behavior
system_message = SystemMessage(content="You are a helpful AI assistant. Answer questions concisely and remember previous conversations.")

# Conversation history
messages = [system_message]

# Create chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("placeholder", "{history}"),
    ("human", "{user_input}")
])

chain = prompt | model

while True:
    user_input = input('You: ')
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. Goodbye!")
        break
    
    messages.append(HumanMessage(content=user_input))
    
    # Format messages for the chain
    history_messages = [f"{msg.__class__.__name__}: {msg.content}" for msg in messages[:-1]]
    history_str = "\n".join(history_messages)
    
    response = chain.invoke({
        "system": system_message.content,
        "history": history_str,
        "user_input": user_input
    }).strip()
    
    # Extract assistant response
    if "Assistant:" in response:
        assistant_response = response.split("Assistant:")[-1].strip()
        if "\nUser:" in assistant_response:
            assistant_response = assistant_response.split("\nUser:")[0].strip()
    else:
        assistant_response = response
    
    messages.append(AIMessage(content=assistant_response))
    print(f'ChatBot: {assistant_response}')
