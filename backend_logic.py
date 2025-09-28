import xarray as xr
import pandas as pd
import psycopg2
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from gtts import gTTS
import os
import uuid
from transformers import pipeline
from indicnlp import common
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

# --- Configuration & Initial Setup ---
INDIC_NLP_RESOURCES_PATH = "indic_nlp_resources"
common.INDIC_RESOURCES_PATH = INDIC_NLP_RESOURCES_PATH
os.makedirs("temp_audio", exist_ok=True) # Directory to store audio responses

# List of Indic languages for special processing
INDIC_LANGS = ['hi', 'ta', 'te', 'kn', 'mr']

# --- Database Connection Details ---
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"

# --- LLM and VectorDB Setup (Initialize once) ---
try:
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    chroma_db = Chroma(
        collection_name="argo_data",
        embedding_function=embeddings,
        persist_directory="chroma_store"
    )
    llm = ChatOllama(model="phi3")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=chroma_db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )
    print("✅ LLM and QA Chain initialized successfully.")
except Exception as e:
    print(f"❌ Error during LLM/VectorDB initialization: {e}")
    qa_chain = None

def setup_database_and_vectors():
    """
    One-time setup to load data, populate PostgreSQL, and create vector embeddings.
    """
    print("🚀 Starting one-time database and vector store setup...")
    # Step 1: Load and process dataset
    try:
        ds = xr.open_dataset("nodc_D1901290_249.nc")
        df = ds.to_dataframe().reset_index()
        temp_col = next((c for c in df.columns if "temp" in c.lower()), None)
        sal_col = next((c for c in df.columns if "psal" in c.lower()), None)

        if not temp_col or not sal_col:
            raise ValueError("Could not detect temperature/salinity fields.")

        df = df[['latitude', 'longitude', 'pres', sal_col, temp_col]].dropna()
        df.rename(columns={sal_col: 'psal', temp_col: 'temp'}, inplace=True)
        print("✅ NetCDF data loaded and processed.")
    except Exception as e:
        print(f"❌ Error processing NetCDF file: {e}")
        return

    # Step 2: Insert data into PostgreSQL
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS argo_data;")
        cur.execute("""
            CREATE TABLE argo_data (
                latitude REAL, longitude REAL, pres REAL, psal REAL, temp REAL
            );
        """)
        rows_to_insert = [tuple(x) for x in df.head(500).to_numpy()] # Insert 500 rows
        insert_query = "INSERT INTO argo_data (latitude, longitude, pres, psal, temp) VALUES (%s, %s, %s, %s, %s)"
        cur.executemany(insert_query, rows_to_insert)
        conn.commit()
        print("✅ Data inserted into PostgreSQL.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"❌ Error with PostgreSQL: {error}")
    finally:
        if conn:
            cur.close()
            conn.close()

    # Step 3: Insert data into ChromaDB for retrieval
    try:
        documents = [
            f"ARGO float at latitude {row.latitude:.2f} and longitude {row.longitude:.2f} "
            f"recorded a temperature of {row.temp:.2f}C and salinity of {row.psal:.2f} "
            f"at a pressure of {row.pres:.1f} decibars."
            for _, row in df.head(100).iterrows() # Embed first 100 rows
        ]
        chroma_db.add_texts(documents)
        print("✅ Data inserted into ChromaDB vector store.")
    except Exception as e:
        print(f"❌ Error inserting data into ChromaDB: {e}")
    print("🎉 Setup complete!")


def process_indic_text(text: str, lang: str) -> str:
    """
    Normalizes and tokenizes text if it's an Indic language.
    """
    if lang in INDIC_LANGS:
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer(lang)
        normalized_text = normalizer.normalize(text)
        tokens = indic_tokenize.trivial_tokenize(normalized_text, lang)
        return " ".join(tokens)
    return text

def text_to_voice(text: str, lang: str) -> str:
    """
    Converts text to speech and saves it as a temporary MP3 file.
    Returns the path to the audio file.
    """
    try:
        tts = gTTS(text, lang=lang)
        filename = f"temp_audio/{uuid.uuid4()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"❌ Error in text-to-speech conversion: {e}")
        return None

def get_answer(query: str, lang: str = 'en') -> tuple[str, str | None]:
    """
    Processes a query, gets an answer from the LLM, and converts it to speech.
    Returns the text answer and the path to the audio file.
    """
    if not qa_chain:
        error_msg = "QA system not initialized."
        return error_msg, text_to_voice(error_msg, lang)

    try:
        # Step 1: Process Indic languages if necessary
        processed_query = process_indic_text(query, lang)
        print(f"🗣️  Processed Query ({lang}):", processed_query)

        # Step 2: Get answer from the QA chain
        answer_obj = qa_chain.invoke({"query": processed_query})
        answer = answer_obj.get("result", "Sorry, I could not find an answer.")
        print("🤖 LLM Answer:", answer)

        # Step 3: Convert the answer to voice
        audio_filepath = text_to_voice(answer, lang)
        
        return answer, audio_filepath

    except Exception as e:
        print(f"❌ An error occurred during query processing: {e}")
        error_response = "I encountered an error while processing your request."
        return error_response, text_to_voice(error_response, lang)