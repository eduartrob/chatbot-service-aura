# app/nlp/prompt_classifier.py
"""
Clasificador de Intención del Prompt.

Detecta la intención del usuario para enrutar apropiadamente:
- Crisis: Requiere intervención urgente
- Support: Busca apoyo emocional
- General: Conversación general
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
import re


class PromptIntent(Enum):
    """Intenciones posibles del prompt del usuario."""
    
    CRISIS = "crisis"           # Situación de crisis, requiere derivación
    SUPPORT = "support"         # Busca apoyo emocional
    INFORMATION = "information" # Busca información
    GREETING = "greeting"       # Saludo
    GENERAL = "general"         # Conversación general


@dataclass
class IntentResult:
    """Resultado de la clasificación de intención."""
    
    intent: PromptIntent
    confidence: float  # 0.0 - 1.0
    matched_patterns: List[str]  # Patrones que activaron la clasificación
    requires_human_intervention: bool = False
    
    @property
    def is_urgent(self) -> bool:
        return self.intent == PromptIntent.CRISIS


class PromptClassifier:
    """
    Clasificador basado en patrones para detectar intención del prompt.
    
    Se enfoca especialmente en detectar situaciones de crisis que
    requieren intervención humana profesional.
    """
    
    # Patrones de crisis (requieren atención inmediata)
    CRISIS_PATTERNS = [
        r'\b(suicid|matarme|quitarme la vida|no quiero vivir)\b',
        r'\b(acabar con todo|terminar con esto)\b',
        r'\b(autolesion|cortar|hacerme daño)\b',
        r'\b(no puedo más|no aguanto más)\b',
        r'\b(quiero morir|deseo morir)\b',
        r'\b(sin salida|no hay esperanza)\b',
    ]
    
    # Patrones de apoyo emocional
    SUPPORT_PATTERNS = [
        r'\b(me siento|siento que)\b',
        r'\b(triste|deprimid|ansios|sol[oa]|vacío)\b',
        r'\b(no sé qué hacer|necesito ayuda|ayúdame)\b',
        r'\b(miedo|preocupad|estresad|agobiad)\b',
        r'\b(nadie me entiende|nadie me quiere)\b',
        r'\b(problemas|dificultades|no puedo)\b',
    ]
    
    # Patrones de saludo
    GREETING_PATTERNS = [
        r'^(hola|hey|buenas|saludos|qué tal|cómo estás)',
        r'^(buenos días|buenas tardes|buenas noches)',
        r'^(hi|hello)\b',
    ]
    
    # Patrones de información
    INFO_PATTERNS = [
        r'\b(qué es|cómo funciona|explica|dime sobre)\b',
        r'\b(información|info|datos)\b',
        r'^(qué|cómo|cuándo|dónde|por qué)\b',
    ]
    
    def __init__(self):
        # Pre-compilar patrones para eficiencia
        self._crisis_regex = [re.compile(p, re.IGNORECASE) for p in self.CRISIS_PATTERNS]
        self._support_regex = [re.compile(p, re.IGNORECASE) for p in self.SUPPORT_PATTERNS]
        self._greeting_regex = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self._info_regex = [re.compile(p, re.IGNORECASE) for p in self.INFO_PATTERNS]
    
    def _match_patterns(self, text: str, patterns: List[re.Pattern]) -> Tuple[int, List[str]]:
        """Cuenta coincidencias de patrones y retorna los matches."""
        matches = []
        for pattern in patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return len(matches), matches
    
    def classify(self, text: str) -> IntentResult:
        """
        Clasifica la intención del prompt.
        
        Args:
            text: Prompt del usuario
            
        Returns:
            IntentResult con la intención detectada
        """
        if not text or len(text.strip()) < 2:
            return IntentResult(
                intent=PromptIntent.GENERAL,
                confidence=0.5,
                matched_patterns=[]
            )
        
        text_lower = text.lower().strip()
        
        # 1. Primero verificar CRISIS (máxima prioridad)
        crisis_count, crisis_matches = self._match_patterns(text_lower, self._crisis_regex)
        if crisis_count > 0:
            return IntentResult(
                intent=PromptIntent.CRISIS,
                confidence=min(0.7 + crisis_count * 0.1, 1.0),
                matched_patterns=crisis_matches,
                requires_human_intervention=True
            )
        
        # 2. Verificar saludo (si es corto y es saludo, clasificar como tal)
        greeting_count, greeting_matches = self._match_patterns(text_lower, self._greeting_regex)
        if greeting_count > 0 and len(text_lower.split()) <= 5:
            return IntentResult(
                intent=PromptIntent.GREETING,
                confidence=0.9,
                matched_patterns=greeting_matches
            )
        
        # 3. Verificar búsqueda de apoyo emocional
        support_count, support_matches = self._match_patterns(text_lower, self._support_regex)
        if support_count >= 1:
            return IntentResult(
                intent=PromptIntent.SUPPORT,
                confidence=min(0.6 + support_count * 0.1, 0.95),
                matched_patterns=support_matches
            )
        
        # 4. Verificar búsqueda de información
        info_count, info_matches = self._match_patterns(text_lower, self._info_regex)
        if info_count > 0:
            return IntentResult(
                intent=PromptIntent.INFORMATION,
                confidence=0.7,
                matched_patterns=info_matches
            )
        
        # 5. Default: conversación general
        return IntentResult(
            intent=PromptIntent.GENERAL,
            confidence=0.5,
            matched_patterns=[]
        )


# Instancia global
_classifier = PromptClassifier()


def get_prompt_classifier() -> PromptClassifier:
    """Obtiene la instancia global del clasificador."""
    return _classifier
