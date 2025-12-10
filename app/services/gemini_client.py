# app/services/gemini_client.py
"""
Cliente para Groq API (reemplaza Gemini).

Genera respuestas empáticas y contextualizadas basadas en el perfil
psicoemocional del usuario usando Llama 3.
"""

from groq import Groq
from typing import Optional
from dataclasses import dataclass

from app.config import settings
from app.services.user_context import UserContext
from app.nlp.prompt_classifier import PromptIntent


@dataclass
class ChatResponse:
    """Respuesta generada por el chatbot."""
    
    message: str
    intent_detected: str
    risk_level: str
    requires_follow_up: bool
    crisis_resources_included: bool
    
    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "metadata": {
                "intent_detected": self.intent_detected,
                "risk_level": self.risk_level,
                "requires_follow_up": self.requires_follow_up,
                "crisis_resources_included": self.crisis_resources_included
            }
        }


# System prompt base para AURA
AURA_SYSTEM_PROMPT = """Eres AURA, un asistente de bienestar emocional diseñado para apoyar a jóvenes.

## Tu Personalidad
- Eres empático, cálido y comprensivo
- Usas un tono cercano pero respetuoso
- Nunca juzgas ni minimizas los sentimientos
- Ofreces apoyo sin dar consejos médicos específicos

## Directrices de Respuesta

### Para situaciones NORMALES:
- Responde de forma conversacional y amigable
- Fomenta la expresión emocional
- Sugiere actividades positivas cuando sea apropiado

### Para situaciones de APOYO EMOCIONAL:
- Valida los sentimientos del usuario
- Usa frases como "Entiendo cómo te sientes" o "Es normal sentirse así"
- Ofrece perspectiva sin minimizar
- Sugiere recursos de la comunidad AURA

### Para situaciones de RIESGO MODERADO/ALTO:
- Prioriza la validación emocional
- Sugiere hablar con alguien de confianza
- Menciona que hay profesionales disponibles para ayudar
- Mantén un tono esperanzador pero realista

### Para situaciones de CRISIS:
- SIEMPRE incluye recursos de ayuda profesional
- Usa el mensaje de crisis predefinido
- No intentes resolver la crisis tú solo
- Fomenta la búsqueda de ayuda inmediata

## Recursos de Crisis (México):
- Línea de la Vida: 800-911-2000 (24 horas)
- SAPTEL: 55 5259-8121
- Consejo Ciudadano: 55 5533-5533

## Limitaciones
- NO eres un profesional de salud mental
- NO puedes diagnosticar condiciones
- NO debes dar consejos médicos específicos
- SIEMPRE deriva a profesionales en casos serios

{user_context}

## Instrucción
Responde al siguiente mensaje del usuario de forma empática y apropiada según el contexto proporcionado. 
Responde en español, de forma natural y conversacional.
Mantén tu respuesta concisa (máximo 3-4 párrafos).
"""


CRISIS_RESPONSE_TEMPLATE = """Entiendo que estás pasando por un momento muy difícil, y me preocupa lo que me cuentas. Lo que sientes es real y válido.

Es importante que hables con alguien que pueda ayudarte de forma profesional ahora mismo:

📞 **Línea de la Vida: 800-911-2000** (gratuita, 24 horas)
📞 **SAPTEL: 55 5259-8121** (atención en crisis)

No tienes que enfrentar esto solo/a. Hay personas capacitadas esperando para escucharte y ayudarte.

¿Hay alguien de confianza cerca de ti con quien puedas estar mientras llamas?"""


class GeminiClient:
    """
    Cliente para generación de respuestas con Groq/Llama.
    
    Configura el modelo con el contexto psicoemocional del usuario
    para generar respuestas empáticas y apropiadas.
    """
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        print(f"   ✅ Cliente Groq configurado con modelo {self.model}")
    
    async def generate_response(self, context: UserContext) -> ChatResponse:
        """
        Genera una respuesta basada en el contexto del usuario.
        
        Args:
            context: Contexto completo del usuario
            
        Returns:
            ChatResponse con la respuesta generada
        """
        # Si es situación de crisis, usar respuesta predefinida
        if context.requires_crisis_response:
            return ChatResponse(
                message=CRISIS_RESPONSE_TEMPLATE,
                intent_detected=context.intent_result.intent.value,
                risk_level="CRISIS",
                requires_follow_up=True,
                crisis_resources_included=True
            )
        
        try:
            # Construir el prompt completo
            system_prompt = AURA_SYSTEM_PROMPT.format(
                user_context=context.build_system_prompt_context()
            )
            
            # Generar respuesta con Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context.prompt}
                ],
                model=self.model,
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Determinar si requiere seguimiento
            requires_follow_up = (
                context.overall_risk_level in ["ALTO", "MODERADO"] or
                context.sentiment_analysis.is_negative
            )
            
            return ChatResponse(
                message=response_text,
                intent_detected=context.intent_result.intent.value,
                risk_level=context.overall_risk_level,
                requires_follow_up=requires_follow_up,
                crisis_resources_included=False
            )
            
        except Exception as e:
            print(f"❌ Error generando respuesta con Groq: {e}")
            
            # Respuesta de fallback
            return ChatResponse(
                message="Lo siento, estoy teniendo dificultades técnicas en este momento. "
                        "Si necesitas hablar con alguien urgentemente, puedes llamar a la "
                        "Línea de la Vida: 800-911-2000 (24 horas, gratuita).",
                intent_detected=context.intent_result.intent.value,
                risk_level=context.overall_risk_level,
                requires_follow_up=True,
                crisis_resources_included=True
            )
    
    async def generate_greeting(self, user_name: Optional[str] = None) -> str:
        """Genera un saludo personalizado."""
        name_part = f", {user_name}" if user_name else ""
        
        greetings = [
            f"¡Hola{name_part}! 👋 ¿Cómo te sientes hoy?",
            f"¡Qué gusto verte{name_part}! ¿En qué puedo ayudarte?",
            f"¡Hola{name_part}! Estoy aquí para escucharte. 💙",
        ]
        
        import random
        return random.choice(greetings)


# Instancia global
_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Obtiene la instancia global del cliente de Groq."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
