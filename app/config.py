# app/config.py
"""Configuración centralizada del servicio de chatbot."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración del servicio."""
    
    # Servicio
    SERVICE_NAME: str = "chatbot-service-aura"
    SERVICE_PORT: int = 8002
    DEBUG: bool = False
    
    # Google Gemini
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-flash-latest"
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_MAX_TOKENS: int = 500
    
    # Clustering Service
    CLUSTERING_SERVICE_URL: str = "http://localhost:8001"
    
    # NLP
    NLP_MODEL_NAME: str = "UMUTeam/roberta-spanish-sentiment-analysis"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Singleton para obtener la configuración."""
    return Settings()


settings = get_settings()
