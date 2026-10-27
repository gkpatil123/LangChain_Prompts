
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 128,
#         "do_sample": False,
#         "return_full_text": False
#     }
# )

# model = ChatHuggingFace(llm=llm)

# chat_history = [
#     SystemMessage(content="You are a helpful AI assistant.")
# ]

# while True:
#     user_input = input("You: ").strip()

#     if user_input.lower() == "exit":
#         print("AI: Goodbye!")
#         break

#     if not user_input:
#         continue

#     chat_history.append(HumanMessage(content=user_input))

#     try:
#         print("AI is thinking...")
#         result = model.invoke(chat_history)

#         chat_history.append(AIMessage(content=result.content))
#         print("AI:", result.content)

#     except Exception as e:
#         chat_history.pop()
#         print("Error:", str(e))

# print("\nChat history:")
# for message in chat_history:
#     print(f"{message.type}: {message.content}")



from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_length": 512,
        "do_sample": False,
        "return_full_text": False
    }
)


model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(
        content="You are a helpful AI assistant. Answer clearly and briefly."
    )
]

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("AI: Goodbye!")
        break

    if not user_input:
        continue

    chat_history.append(HumanMessage(content=user_input))

    try:
        print("AI is thinking...")
        result = model.invoke(chat_history)
        answer = result.content.strip()

        chat_history.append(AIMessage(content=answer))
        print("AI:", answer)

    except Exception as e:
        chat_history.pop()
        print("Error:", e)

print("\nChat history:")
for message in chat_history:
    print(f"{message.type}: {message.content}")
