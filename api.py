# api.py (MODIFIED)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager

# Import functions from your backend logic
from backend_logic import get_answer, setup_database_and_vectors

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs on startup
    print("Server starting up...")
    if not os.path.exists("chroma_store"): # Run setup only if it hasn't been run
        setup_database_and_vectors()
    else:
        print("✅ Vector store already exists. Skipping setup.")
    yield
    # This runs on shutdown
    print("Server shutting down...")
    # Clean up temp audio files if any
    for f in os.listdir("temp_audio"):
        os.remove(os.path.join("temp_audio", f))


app = FastAPI(lifespan=lifespan)

# Mount a directory to serve the generated audio files
app.mount("/audio", StaticFiles(directory="temp_audio"), name="audio")

class QueryRequest(BaseModel):
    query: str
    lang: str = 'en'

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
async def handle_query(request: QueryRequest):
    """
    Receives a text query and language, gets an answer, and returns
    the text response and the LOCAL FILE PATH to the audio response.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    text_answer, audio_path = get_answer(request.query, request.lang)

    # --- MODIFICATION ---
    # Instead of creating a URL, we now return the direct file path.
    # The Gradio Audio component can handle a local file path directly.
    return {"text_response": text_answer, "audio_path": audio_path}
    # --- END MODIFICATION ---

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serves a specific audio file from the temp directory.
    (This is no longer used by the frontend but is good to keep for debugging)
    """
    file_path = os.path.join("temp_audio", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
    raise HTTPException(status_code=404, detail="Audio file not found.")