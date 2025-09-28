# frontend.py (MODIFIED)

import gradio as gr
import requests
import speech_recognition as sr

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
QUERY_URL = f"{API_BASE_URL}/query"

SUPPORTED_LANGS = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn',
    'Marathi': 'mr'
}

# --- Backend Communication ---
def get_chatbot_response(message, lang_code):
    """Sends the user's message and language to the backend API."""
    try:
        response = requests.post(QUERY_URL, json={"query": message, "lang": lang_code})
        response.raise_for_status()
        data = response.json()
        # --- MODIFICATION ---
        # We now get 'audio_path' directly from the API response
        audio_path = data.get("audio_path")
        return data["text_response"], audio_path
        # --- END MODIFICATION ---
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return f"Error connecting to the backend: {e}", None

def transcribe_audio(filepath):
    """Converts an audio file to text."""
    if filepath is None:
        return ""
    recognizer = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_sphinx(audio_data)
            print(f"Transcribed Text: {text}")
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Recognition service error; {e}"

def process_input(message_or_filepath, chat_history, language):
    """Handles new user input from either text box or audio file."""
    if isinstance(message_or_filepath, str) and message_or_filepath.endswith(('.wav', '.mp3')):
        message = transcribe_audio(message_or_filepath)
    else:
        message = message_or_filepath

    if not message:
        return chat_history, "", None

    lang_code = SUPPORTED_LANGS.get(language, 'en')
    # --- MODIFICATION ---
    # The function now returns response_text and audio_path
    response_text, audio_path = get_chatbot_response(message, lang_code)
    # --- END MODIFICATION ---
    
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response_text})
    
    # --- MODIFICATION ---
    # We pass the local audio_path directly to the audio output component
    return chat_history, "", audio_path
    # --- END MODIFICATION ---

# --- Build Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="ARGO Float Chatbot") as demo:
    gr.Markdown("# 🌊 ARGO Float Chatbot")

    chatbot = gr.Chatbot(label="Conversation", height=550, type="messages")
    
    with gr.Row():
        audio_output = gr.Audio(label="Chatbot Response", autoplay=True, interactive=False)
        lang_dropdown = gr.Dropdown(choices=list(SUPPORTED_LANGS.keys()), value="English", label="Select Language")

    with gr.Row():
        text_input = gr.Textbox(label="Type your question", placeholder="Press Enter to send...", scale=4)
        mic_input = gr.Audio(
            label="Speak your question",
            sources=["microphone"], 
            type="filepath",
            scale=1
        )

    text_input.submit(fn=process_input, inputs=[text_input, chatbot, lang_dropdown], outputs=[chatbot, text_input, audio_output])
    mic_input.change(fn=process_input, inputs=[mic_input, chatbot, lang_dropdown], outputs=[chatbot, text_input, audio_output])

if __name__ == "__main__":
    demo.launch()