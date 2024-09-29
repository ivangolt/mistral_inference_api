import os

import gradio as gr
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


# Mistral API setup
api_key = os.getenv("MISTRAL_API_KEY")
model = os.getenv("MODEL_TYPE")
client = Mistral(api_key=api_key)


# Function to interact with Mistral API
def get_chat_response(user_input: str):
    """
    Interact with Mistral API to get a chat response.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The response message from the Mistral AI.
    """

    chat_response = client.chat.complete(
        model=model, messages=[{"role": "user", "content": user_input}]
    )
    return chat_response.choices[0].message.content


# Gradio interface setup
def mistral_chat(user_input: str):
    """
    Use the Mistral API to get a chat response.

    Args:
        user_input (str): The user's input message.

    Returns:
        str: The response message from the Mistral AI.
    """

    response = get_chat_response(user_input)
    return response


# Define Gradio input and output interface
iface = gr.Interface(
    fn=mistral_chat,
    inputs="text",
    outputs="text",
    title="Mistral AI Chat",
    description="Chat with the Mistral model using an API",
)

# Launch the Gradio app
iface.launch(share=True)
