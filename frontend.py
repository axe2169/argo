import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/query"

def chat_interface(message, history):
    response = requests.post(API_URL, json={"query": message})
    return response.json()["response"]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌊 ARGO Float Chatbot
        Explore the world's oceans using natural language queries.
        Ask questions about temperature, salinity, and float locations.
        """
    )
    
    chatbot = gr.ChatInterface(
        fn=chat_interface,
        title="ARGO Chatbot",
        description="Ask me anything about ARGO float data in the Indian Ocean.",
        examples=[
            "Show me the average temperature in the last 3 months",
            "What is the salinity at a depth of 50m?",
            "Compare temperature and salinity near the equator"
        ]
    )

demo.launch()