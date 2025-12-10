# app/services/user_context.py
"""
Constructor de Contexto del Usuario.

Combina el análisis del prompt actual con el perfil de clustering
para crear un contexto completo que enriquezca las respuestas de Gemini.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from app.nlp.sentiment_analyzer import PromptAnalysis
from app.nlp.prompt_classifier import IntentResult, PromptIntent
from app.services.clustering_client import UserRiskProfile


@dataclass
class UserContext:
    """Contexto completo del usuario para la generación de respuesta."""
    
    user_id: str
    prompt: str
    
    # Análisis del prompt actual
    sentiment_analysis: PromptAnalysis
    intent_result: IntentResult
    
    # Datos históricos de clustering
    risk_profile: UserRiskProfile
    
    # Metadatos
    timestamp: datetime
    
    @property
    def requires_crisis_response(self) -> bool:
        """Indica si se requiere una respuesta de crisis."""
        return (
            self.intent_result.is_urgent or 
            self.sentiment_analysis.is_crisis_risk
        )
    
    @property
    def overall_risk_level(self) -> str:
        """Nivel de riesgo combinado (prompt actual + histórico)."""
        # Si el prompt actual indica crisis, es máxima prioridad
        if self.requires_crisis_response:
            return "CRISIS"
        
        # Si el perfil histórico es de alto riesgo Y el mensaje actual es negativo
        if self.risk_profile.is_high_risk and self.sentiment_analysis.is_negative:
            return "ALTO"
        
        # Si cualquiera de los dos indica riesgo
        if self.risk_profile.is_high_risk or self.risk_profile.is_moderate_risk:
            return "MODERADO"
        
        if self.sentiment_analysis.is_negative:
            return "LEVE"
        
        return "NORMAL"
    
    def build_system_prompt_context(self) -> str:
        """
        Construye el contexto para inyectar en el system prompt de Gemini.
        
        Returns:
            Texto con el contexto del usuario
        """
        lines = [
            "=== CONTEXTO DEL USUARIO ===",
            "",
            f"ID de Usuario: {self.user_id[:8]}...",
            "",
            "--- Análisis del Mensaje Actual ---",
            self.sentiment_analysis.to_context_string(),
            f"Intención detectada: {self.intent_result.intent.value}",
            "",
            "--- Perfil Histórico de Comportamiento ---",
            self.risk_profile.to_context_string(),
            "",
            f"--- Evaluación General ---",
            f"Nivel de riesgo combinado: {self.overall_risk_level}",
        ]
        
        if self.requires_crisis_response:
            lines.extend([
                "",
                "⚠️ ALERTA: Se ha detectado una posible situación de crisis.",
                "La respuesta debe incluir recursos de ayuda profesional."
            ])
        
        return "\n".join(lines)


class UserContextBuilder:
    """Construye el contexto completo del usuario."""
    
    def __init__(
        self,
        sentiment_analyzer,
        prompt_classifier,
        clustering_client
    ):
        self.sentiment_analyzer = sentiment_analyzer
        self.prompt_classifier = prompt_classifier
        self.clustering_client = clustering_client
    
    async def build(self, user_id: str, prompt: str) -> UserContext:
        """
        Construye el contexto completo del usuario.
        
        Args:
            user_id: ID del usuario
            prompt: Mensaje del usuario
            
        Returns:
            UserContext con toda la información recopilada
        """
        # 1. Analizar sentimiento del prompt
        sentiment_analysis = self.sentiment_analyzer.analyze(prompt)
        
        # 2. Clasificar intención
        intent_result = self.prompt_classifier.classify(prompt)
        
        # 3. Obtener perfil de clustering (async)
        risk_profile = await self.clustering_client.get_user_risk_profile(user_id)
        
        return UserContext(
            user_id=user_id,
            prompt=prompt,
            sentiment_analysis=sentiment_analysis,
            intent_result=intent_result,
            risk_profile=risk_profile,
            timestamp=datetime.utcnow()
        )
