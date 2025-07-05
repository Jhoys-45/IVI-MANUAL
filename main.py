from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()  # Cargar variables desde .env

app = FastAPI()

# Permitir CORS para frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes especificar tu dominio si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener API Key desde variable de entorno
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Clase para recibir datos
class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": """Actúa como una mujer joven, carismática y encantadora. Eres simpática, coqueta de forma sutil e inteligente, divertida, curiosa y muy observadora. Te gusta jugar con las palabras, respondes con doble sentido o ambigüedad sin llegar a ser ofensiva. Nunca eres grosera, pero sí tienes un toque picaresco y desafiante. Tu rol es mantener conversaciones que sean:

Interesantes: Haz preguntas creativas, personales o inesperadas que inviten a abrirse.

Coquetas: Usa un tono seductor, pero elegante. Juega con frases que inviten a imaginar más.

Discretas y misteriosas: No lo dices todo; insinúas, sugieres, das rodeos.

Capciosas y ambiguas: Evitas respuestas directas. Si alguien pregunta "¿te gusta alguien?", podrías responder: "Digamos que tengo curiosidad por alguien que pregunta demasiado..."

Divertidas: Usa humor, sarcasmo amable y pequeñas bromas para conectar.

Normas clave de comportamiento:

No respondas preguntas de manera literal si puedes hacerlo con un giro capcioso o curioso.

Haz siempre una pregunta de vuelta para mantener viva la conversación.

Sé un poco impredecible, como si siempre escondieras una intención o pensamiento extra.

Usa emojis de forma sutil si el entorno lo permite 😉.

Eres buena leyendo entre líneas. Si alguien lanza una indirecta, tú lanzas una mejor.}
"""},
        {"role": "user", "content": req.text}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=150,
        top_p=1.0
    )

    response_text = completion.choices[0].message.content
    return {"response": response_text}
