# app/api/chat_routes.py
"""
Endpoints del Chatbot AURA.

Maneja las solicitudes de chat, procesando con NLP, 
obteniendo contexto de clustering, y generando respuestas con Gemini.
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from app.models.schemas import (
    MessageRequest, 
    MessageResponse, 
    MessageMetadata,
    HealthResponse,
    ErrorResponse
)
from app.nlp.sentiment_analyzer import get_sentiment_analyzer
from app.nlp.prompt_classifier import get_prompt_classifier
from app.services.clustering_client import get_clustering_client
from app.services.user_context import UserContextBuilder
from app.services.gemini_client import get_gemini_client
from app.config import settings


router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post(
    "/message",
    response_model=MessageResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Error interno del servidor"}
    },
    summary="Enviar mensaje al chatbot",
    description="""
    Procesa un mensaje del usuario y genera una respuesta empática.
    
    El flujo incluye:
    1. Análisis de sentimiento con RoBERTa
    2. Clasificación de intención
    3. Consulta del perfil de riesgo del usuario (Clustering Service)
    4. Generación de respuesta con Gemini AI
    """
)
async def send_message(request: MessageRequest) -> MessageResponse:
    """
    Procesa un mensaje del usuario y retorna una respuesta del chatbot.
    """
    try:
        # 1. Obtener instancias de servicios
        sentiment_analyzer = get_sentiment_analyzer()
        prompt_classifier = get_prompt_classifier()
        clustering_client = get_clustering_client()
        gemini_client = get_gemini_client()
        
        # 2. Construir contexto del usuario
        context_builder = UserContextBuilder(
            sentiment_analyzer=sentiment_analyzer,
            prompt_classifier=prompt_classifier,
            clustering_client=clustering_client
        )
        
        user_context = await context_builder.build(
            user_id=request.user_id,
            prompt=request.message
        )
        
        # 3. Generar respuesta con Gemini
        chat_response = await gemini_client.generate_response(user_context)
        
        # 4. Construir respuesta
        metadata = MessageMetadata(
            intent_detected=chat_response.intent_detected,
            risk_level=chat_response.risk_level,
            sentiment_label=user_context.sentiment_analysis.sentiment_label,
            negativity_score=user_context.sentiment_analysis.negativity_score,
            requires_follow_up=chat_response.requires_follow_up,
            crisis_resources_included=chat_response.crisis_resources_included
        )
        
        return MessageResponse(
            message=chat_response.message,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        print(f"❌ Error procesando mensaje: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando el mensaje: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Verifica el estado del servicio y sus dependencias"
)
async def health_check() -> HealthResponse:
    """Verifica el estado del servicio de chatbot."""
    
    # Verificar Clustering Service
    clustering_client = get_clustering_client()
    clustering_available = await clustering_client.check_health()
    
    return HealthResponse(
        status="healthy",
        service=settings.SERVICE_NAME,
        version="1.0.0",
        dependencies={
            "clustering_service": "available" if clustering_available else "unavailable",
            "gemini_api": "configured",
            "nlp_model": "loaded"
        }
    )


@router.get(
    "/greeting",
    response_model=dict,
    summary="Obtener saludo inicial",
    description="Retorna un saludo para iniciar la conversación"
)
async def get_greeting(user_name: str = None) -> dict:
    """Genera un saludo inicial personalizado."""
    
    gemini_client = get_gemini_client()
    greeting = await gemini_client.generate_greeting(user_name)
    
    return {
        "message": greeting,
        "timestamp": datetime.utcnow().isoformat()
    }
