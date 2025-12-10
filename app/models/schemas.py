# app/models/schemas.py
"""
Esquemas Pydantic para validación de request/response.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class MessageRequest(BaseModel):
    """Request para enviar un mensaje al chatbot."""
    
    user_id: str = Field(
        ..., 
        description="UUID del usuario (de Auth Service)",
        min_length=1
    )
    message: str = Field(
        ..., 
        description="Mensaje del usuario",
        min_length=1,
        max_length=2000
    )
    session_id: Optional[str] = Field(
        None,
        description="ID de sesión para mantener contexto de conversación"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Me siento muy solo últimamente",
                "session_id": "session-123"
            }
        }


class MessageMetadata(BaseModel):
    """Metadatos del análisis del mensaje."""
    
    intent_detected: str = Field(..., description="Intención detectada del mensaje")
    risk_level: str = Field(..., description="Nivel de riesgo evaluado")
    sentiment_label: str = Field(..., description="Etiqueta de sentimiento (POS/NEG/NEU)")
    negativity_score: float = Field(..., ge=0, le=1, description="Score de negatividad")
    requires_follow_up: bool = Field(..., description="Si requiere seguimiento")
    crisis_resources_included: bool = Field(..., description="Si se incluyeron recursos de crisis")


class MessageResponse(BaseModel):
    """Response del chatbot."""
    
    message: str = Field(..., description="Respuesta generada por el chatbot")
    metadata: MessageMetadata = Field(..., description="Metadatos del análisis")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Entiendo cómo te sientes. Es muy válido sentirse solo a veces...",
                "metadata": {
                    "intent_detected": "support",
                    "risk_level": "MODERADO",
                    "sentiment_label": "NEG",
                    "negativity_score": 0.65,
                    "requires_follow_up": True,
                    "crisis_resources_included": False
                },
                "timestamp": "2025-12-09T15:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Response del health check."""
    
    status: str
    service: str
    version: str
    dependencies: dict


class ErrorResponse(BaseModel):
    """Response de error."""
    
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
