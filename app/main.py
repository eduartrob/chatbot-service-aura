# app/main.py
"""
Punto de entrada principal del Chatbot Service AURA.

Servicio de IA conversacional para apoyo psicoemocional que integra:
- Análisis de sentimiento con RoBERTa
- Perfiles de riesgo del Clustering Service
- Generación de respuestas con Gemini AI
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from app.config import settings
from app.api.chat_routes import router as chat_router

# Cargar variables de entorno
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hook de ciclo de vida: inicialización y cierre del servicio."""
    
    # === STARTUP ===
    print("\n" + "="*60)
    print(f"🤖 Iniciando {settings.SERVICE_NAME}...")
    print("="*60)
    
    # Pre-cargar modelo NLP (opcional, se cargará en primer uso si no)
    try:
        from app.nlp.sentiment_analyzer import get_sentiment_analyzer
        analyzer = get_sentiment_analyzer()
        # Forzar carga del modelo
        _ = analyzer.sentiment_pipeline
        print("   ✅ Modelo NLP (RoBERTa) cargado")
    except Exception as e:
        print(f"   ⚠️ Modelo NLP se cargará en primer uso: {e}")
    
    # Verificar configuración de Gemini
    try:
        from app.services.gemini_client import get_gemini_client
        _ = get_gemini_client()
        print("   ✅ Cliente Gemini configurado")
    except Exception as e:
        print(f"   ❌ Error configurando Gemini: {e}")
    
    # Verificar conexión con Clustering Service
    try:
        from app.services.clustering_client import get_clustering_client
        client = get_clustering_client()
        is_available = await client.check_health()
        if is_available:
            print("   ✅ Clustering Service disponible")
        else:
            print("   ⚠️ Clustering Service no disponible (continuando sin perfil)")
    except Exception as e:
        print(f"   ⚠️ No se pudo verificar Clustering Service: {e}")
    
    print(f"\n   ✅ Servicio listo en puerto {settings.SERVICE_PORT}")
    print(f"   📚 Documentación: http://localhost:{settings.SERVICE_PORT}/docs")
    print("="*60 + "\n")
    
    yield
    
    # === SHUTDOWN ===
    print(f"\n👋 Cerrando {settings.SERVICE_NAME}...")


# Inicializar aplicación FastAPI
app = FastAPI(
    title="AURA Chatbot API",
    description="""
## 🤖 API de Chatbot para Apoyo Psicoemocional - AURA

Este servicio proporciona un asistente conversacional de IA diseñado para
ofrecer apoyo emocional a jóvenes usuarios de la plataforma AURA.

### 🧠 Características

* **Análisis de Sentimiento:** RoBERTa pre-entrenado para español
* **Detección de Crisis:** Identificación automática de situaciones urgentes
* **Contexto de Usuario:** Integración con perfil de riesgo del Clustering Service
* **Respuestas Empáticas:** Generación con Gemini AI y directrices psicoemocionales

### 🔗 Endpoints

* `POST /api/v1/chat/message` - Enviar mensaje y recibir respuesta
* `GET /api/v1/chat/health` - Estado del servicio
* `GET /api/v1/chat/greeting` - Saludo inicial

### ⚠️ Importante

Este chatbot NO reemplaza la atención profesional de salud mental.
En situaciones de crisis, siempre se derivan recursos de ayuda profesional.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(chat_router)


@app.get("/", tags=["Root"])
def root():
    """Endpoint raíz con información del servicio."""
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "description": "AURA Chatbot - Asistente de Apoyo Psicoemocional",
        "documentation": "/docs",
        "endpoints": {
            "chat": "/api/v1/chat/message",
            "health": "/api/v1/chat/health",
            "greeting": "/api/v1/chat/greeting"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    """Health check básico."""
    return {"status": "healthy"}


# Punto de entrada para ejecución directa
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG
    )
