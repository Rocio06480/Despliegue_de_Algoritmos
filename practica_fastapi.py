from fastapi import FastAPI
from transformers import pipeline

app = FastAPI(title="Práctica parte FastAPI")


sentiment_pipe = pipeline("sentiment-analysis")
generator_pipe = pipeline("text-generation", model="gpt2")


@app.get("/")
def read_root():
    return {"mensaje": "Hola, bienvenido a este modulo"}


@app.get("/quien_soy")
def read_info():
    return {
        "alumno": "Rocío",
        "proyecto": "Practica parte FastAPI",
        "estado": "Activo"}


@app.get("/health")
def read_health():
    return {"estado_sistema": "Operativo"}


@app.get("/ia_sentimiento")
def get_sentiment(texto: str = "I am happy with this project"):
    resultado = sentiment_pipe(texto)
    return {"texto_original": texto, "resultado_ia": resultado}


@app.get("/ia_completar")
def get_completion(frase: str = "The future of AI is"):
    resultado = generator_pipe(frase, max_length=20, num_return_sequences=1)
    return {"frase_entrada": frase, "texto_generado": resultado[0]['generated_text']}
