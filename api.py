from fastapi import FastAPI
from pydantic import BaseModel
from backend_logic import get_answer
import pandas as pd
import folium
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text
import urllib.parse

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
async def handle_query(request: QueryRequest):
    answer = get_answer(request.query)
    return {"response": answer}

@app.get("/map", response_class=HTMLResponse)
async def get_map():
    # PostgreSQL connection details for map data
    DB_HOST = "localhost"
    DB_NAME = "postgres"
    DB_USER = "postgres"
    DB_PASS = "postgres"
    DB_PORT = "5433"
    db_url = f"postgresql://{DB_USER}:{urllib.parse.quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    with engine.connect() as connection:
        query = text("SELECT * FROM argo_data LIMIT 50")
        df = pd.read_sql_query(query, connection)

    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=2)
    for index, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Temp: {row['temperature']:.2f}, Sal: {row['salinity']:.2f}"
        ).add_to(m)
    return m._repr_html_()