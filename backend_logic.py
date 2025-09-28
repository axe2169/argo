import xarray as xr
import pandas as pd
import psycopg2
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import torch
from transformers import pipeline
from indicnlp import common
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from langdetect import detect as old_detect

# --- Initial setup for IndicNLP resources ---
INDIC_NLP_RESOURCES_PATH = "indic_nlp_resources"
common.INDIC_RESOURCES_PATH = INDIC_NLP_RESOURCES_PATH

# Supported Indian languages and their codes for gTTS and internal logic
SUPPORTED_LANGS = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'mr': 'Marathi'
}
# A separate list for Indic languages to control processing
INDIC_LANGS = ['hi', 'ta', 'te', 'kn', 'mr']

# --- New: Language detection pipeline from Hugging Face ---
if torch.cuda.is_available():
    device = 0
else:
    device = -1

lang_pipeline = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=device
)

def process_indic_text(text: str, lang: str) -> str:
    if lang in INDIC_LANGS:
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer(lang)
        normalized_text = normalizer.normalize(text)
        tokens = indic_tokenize.trivial_tokenize(normalized_text, lang)
        return " ".join(tokens)
    return text

def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak your query...")
        audio = r.listen(source)
    
    try:
        text = r.recognize_google(audio, language='auto')
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

def text_to_voice(text, lang='en'):
    tts = gTTS(text, lang=lang)
    tts.save("chatbot_response.mp3")
    playsound("chatbot_response.mp3")
    os.remove("chatbot_response.mp3")

# --- PostgreSQL Connection Details ---
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"

# --- Step 1: Load dataset ---
ds = xr.open_dataset("pythonProject1/nodc_D1901290_249.nc")

# --- Step 2: Convert to DataFrame ---
df = ds.to_dataframe().reset_index()
print("\n✅ Converted NetCDF to DataFrame")

# --- Step 3: Detect temperature & salinity columns automatically ---
temp_vars = [c for c in df.columns if "temp" in c.lower()]
sal_vars = [c for c in df.columns if "sal" in c.lower()]

if temp_vars and sal_vars:
    temp_col = temp_vars[0]
    sal_col = sal_vars[0]
    df = df.dropna(subset=[temp_col, sal_col])
    print(f"\n✅ Using {temp_col} and {sal_col} for cleaning")
else:
    raise ValueError("❌ Could not detect temperature/salinity fields in dataset")

# --- Step 4: Insert Data into PostgreSQL ---
conn = None
try:
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='postgres',
        host='localhost',
        port='5433'
    )
    cur = conn.cursor()
    
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS argo_data (
            latitude REAL,
            longitude REAL,
            pres REAL,
            psal REAL,
            temp REAL
        );
    """)

    rows_to_insert = [
        (row['latitude'], row['longitude'], row['pres'], row['psal'], row['temp'])
        for _, row in df.head(10).iterrows()
    ]
    
    cur.executemany(f"""
        INSERT INTO argo_data (latitude, longitude, pres, psal, temp)
        VALUES (%s, %s, %s, %s, %s);
    """, rows_to_insert)

    conn.commit()
    print("\n🎉 Done! Data inserted into PostgreSQL")

except (Exception, psycopg2.DatabaseError) as error:
    print(f"\n❌ Error inserting data into PostgreSQL: {error}")
finally:
    if conn:
        cur.close()
        conn.close()

# --- Step 5: Setup Embeddings & ChromaDB ---
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

chroma_db = Chroma(
    collection_name="argo_data",
    embedding_function=embeddings,
    persist_directory="chroma_store"
)

# --- Step 6: Insert Sample Data into Chroma ---
documents = []
for _, row in df.head(5).iterrows():
    doc = f"""
    ARGO float at ({row.get('latitude', 'NA')}, {row.get('longitude', 'NA')}),
    pressure {row.get('pres', 'NA')}m recorded temperature {row[temp_col]}
    and salinity {row[sal_col]}.
    """
    documents.append(doc)

chroma_db.add_texts(documents)
print("\n✅ Inserted ARGO rows into ChromaDB")

# --- Step 7: Chatbot setup ---
llm = ChatOllama(model="phi3")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=chroma_db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# --- Step 8: Get query from voice input ---
user_query = voice_to_text()

# --- Step 9: Process query, detect language, and get answer from chatbot ---
try:
    results = lang_pipeline(user_query)
    detected_lang = results[0]['label']
    print(f"Detected language: {SUPPORTED_LANGS.get(detected_lang, 'Unknown')}")
    
    if detected_lang not in SUPPORTED_LANGS:
        print("❌ This chatbot only supports Indian languages and English.")
        text_to_voice("I'm sorry, I can only respond in Indian languages or English.")
        exit()

    processed_query = process_indic_text(user_query, lang=detected_lang)
    answer_obj = qa_chain.invoke({"query": processed_query})
    answer = answer_obj.get("result", "")
    print("\n🤖 Chatbot:", answer)

    text_to_voice(answer, lang=detected_lang)

except Exception as e:
    print(f"An error occurred: {e}")