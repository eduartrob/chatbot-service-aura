# app/services/clustering_client.py
"""
Cliente HTTP para el Servicio de Clustering.

Obtiene el perfil de riesgo del usuario desde el Clustering Service API v2
para enriquecer el contexto de las respuestas del chatbot.
"""

import httpx
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

from app.config import settings


@dataclass
class UserRiskProfile:
    """Perfil de riesgo del usuario obtenido del Clustering Service."""
    
    user_id: str
    risk_level: str  # ALTO_RIESGO, RIESGO_MODERADO, BAJO_RIESGO
    severity_index: float  # 0-100
    
    # KPIs individuales (0-1 normalizados)
    inactivity_score: float
    night_activity_score: float
    negativity_score: float
    community_engagement: float
    
    last_updated: Optional[datetime] = None
    
    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == "ALTO_RIESGO"
    
    @property
    def is_moderate_risk(self) -> bool:
        return self.risk_level == "RIESGO_MODERADO"
    
    def get_risk_factors(self) -> list[str]:
        """Retorna una lista de factores de riesgo detectados."""
        factors = []
        
        if self.inactivity_score > 0.6:
            factors.append("Inactividad prolongada en la plataforma")
        if self.night_activity_score > 0.5:
            factors.append("Patrón de actividad nocturna elevado")
        if self.negativity_score > 0.5:
            factors.append("Contenido con tono emocional negativo")
        if self.community_engagement < 0.3:
            factors.append("Baja participación en comunidades")
        
        return factors
    
    def to_context_string(self) -> str:
        """Convierte el perfil a texto para el contexto de Gemini."""
        risk_desc = {
            "ALTO_RIESGO": "alto riesgo psicoemocional",
            "RIESGO_MODERADO": "riesgo moderado",
            "BAJO_RIESGO": "bajo riesgo"
        }.get(self.risk_level, "desconocido")
        
        factors = self.get_risk_factors()
        factors_text = ", ".join(factors) if factors else "sin factores significativos"
        
        return (
            f"Perfil del usuario: {risk_desc} (severidad: {self.severity_index:.0f}/100). "
            f"Factores observados: {factors_text}."
        )
    
    @classmethod
    def default(cls, user_id: str) -> 'UserRiskProfile':
        """Crea un perfil por defecto cuando no hay datos disponibles."""
        return cls(
            user_id=user_id,
            risk_level="DESCONOCIDO",
            severity_index=0.0,
            inactivity_score=0.0,
            night_activity_score=0.0,
            negativity_score=0.0,
            community_engagement=0.5
        )


class ClusteringClient:
    """
    Cliente HTTP asíncrono para el Clustering Service.
    
    Obtiene datos del perfil de riesgo del usuario para contextualizar
    las respuestas del chatbot.
    """
    
    def __init__(self):
        self.base_url = settings.CLUSTERING_SERVICE_URL
        self.timeout = 10.0  # segundos
    
    async def get_user_risk_profile(self, user_id: str) -> UserRiskProfile:
        """
        Obtiene el perfil de riesgo de un usuario.
        
        Args:
            user_id: UUID del usuario
            
        Returns:
            UserRiskProfile con los datos de riesgo
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Intentar obtener del endpoint de usuarios de alto riesgo primero
                response = await client.get(
                    f"{self.base_url}/api/v2/clustering/data/high-risk-users",
                    params={"limit": 50}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Buscar el usuario en la lista
                    for user in data.get("users", []):
                        if user.get("user_id") == user_id:
                            factors = user.get("factors", {})
                            return UserRiskProfile(
                                user_id=user_id,
                                risk_level=user.get("risk_level", "DESCONOCIDO"),
                                severity_index=user.get("severity_index", 0),
                                inactivity_score=factors.get("inactivity", 0) / 100,
                                night_activity_score=factors.get("night_activity", 0) / 100,
                                negativity_score=factors.get("negativity", 0) / 100,
                                community_engagement=factors.get("community_engagement", 50) / 100,
                                last_updated=datetime.fromisoformat(user["last_updated"]) if user.get("last_updated") else None
                            )
                
                # Si no está en alto riesgo, asumir bajo riesgo
                return UserRiskProfile(
                    user_id=user_id,
                    risk_level="BAJO_RIESGO",
                    severity_index=20.0,
                    inactivity_score=0.2,
                    night_activity_score=0.1,
                    negativity_score=0.2,
                    community_engagement=0.6
                )
                
        except httpx.TimeoutException:
            print(f"⚠️ Timeout al conectar con Clustering Service")
            return UserRiskProfile.default(user_id)
            
        except httpx.RequestError as e:
            print(f"⚠️ Error de conexión con Clustering Service: {e}")
            return UserRiskProfile.default(user_id)
            
        except Exception as e:
            print(f"⚠️ Error inesperado obteniendo perfil de riesgo: {e}")
            return UserRiskProfile.default(user_id)
    
    async def check_health(self) -> bool:
        """Verifica si el Clustering Service está disponible."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False


# Instancia global
_client: Optional[ClusteringClient] = None


def get_clustering_client() -> ClusteringClient:
    """Obtiene la instancia global del cliente de clustering."""
    global _client
    if _client is None:
        _client = ClusteringClient()
    return _client
