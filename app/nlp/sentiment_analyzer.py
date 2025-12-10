# app/nlp/sentiment_analyzer.py
"""
Analizador de Sentimiento con RoBERTa para Español.

Este módulo evalúa el tono emocional de los mensajes del usuario para
contextualizar las respuestas del chatbot.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dataclasses import dataclass
from typing import Optional
import torch

from app.config import settings


@dataclass
class PromptAnalysis:
    """Resultado del análisis de sentimiento de un prompt."""
    
    text: str
    sentiment_label: str  # 'POS', 'NEG', 'NEU'
    negativity_score: float  # 0.0 - 1.0
    positivity_score: float  # 0.0 - 1.0
    emotional_intensity: float  # 0.0 - 1.0, qué tan fuerte es la emoción
    
    @property
    def is_negative(self) -> bool:
        """Indica si el prompt tiene tono predominantemente negativo."""
        return self.negativity_score > 0.5
    
    @property
    def is_crisis_risk(self) -> bool:
        """Indica si podría ser una situación de crisis (muy negativo + alta intensidad)."""
        return self.negativity_score > 0.7 and self.emotional_intensity > 0.6
    
    def to_context_string(self) -> str:
        """Convierte el análisis a texto para incluir en el contexto de Gemini."""
        intensity_desc = "alta" if self.emotional_intensity > 0.6 else "moderada" if self.emotional_intensity > 0.3 else "baja"
        tone_desc = "negativo" if self.is_negative else "positivo" if self.positivity_score > 0.5 else "neutro"
        
        return f"Tono emocional del mensaje: {tone_desc} (intensidad {intensity_desc}, negatividad: {self.negativity_score:.0%})"


class SentimentAnalyzer:
    """
    Analizador de sentimiento usando RoBERTa pre-entrenado para español.
    
    Utiliza el mismo modelo que el servicio de clustering para consistencia
    en la evaluación psicoemocional de los usuarios.
    """
    
    def __init__(self):
        self.model_name = settings.NLP_MODEL_NAME
        self.device = 0 if torch.cuda.is_available() else -1
        self._pipeline = None
        self._tokenizer = None
    
    @property
    def sentiment_pipeline(self):
        """Inicialización lazy del pipeline de sentimiento."""
        if self._pipeline is None:
            print(f"🔄 Cargando modelo NLP: {self.model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                return_all_scores=True,
                truncation=True,
                max_length=512
            )
            self._tokenizer = tokenizer
            
            print("   ✅ Modelo NLP cargado correctamente")
        
        return self._pipeline
    
    def analyze(self, text: str) -> PromptAnalysis:
        """
        Analiza el sentimiento de un texto.
        
        Args:
            text: Texto a analizar (prompt del usuario)
            
        Returns:
            PromptAnalysis con métricas de sentimiento
        """
        if not text or len(text.strip()) < 2:
            return PromptAnalysis(
                text=text or "",
                sentiment_label="NEU",
                negativity_score=0.0,
                positivity_score=0.0,
                emotional_intensity=0.0
            )
        
        # Truncar si es muy largo
        text_truncated = text[:512]
        
        try:
            result = self.sentiment_pipeline(text_truncated)[0]
            
            # Extraer scores por etiqueta
            scores = {item['label'].upper(): item['score'] for item in result}
            
            # Mapear etiquetas del modelo (pueden variar)
            neg_labels = ['NEG', 'NEGATIVE', 'LABEL_0']
            pos_labels = ['POS', 'POSITIVE', 'LABEL_2']
            neu_labels = ['NEU', 'NEUTRAL', 'LABEL_1']
            
            neg_score = max([scores.get(l, 0) for l in neg_labels])
            pos_score = max([scores.get(l, 0) for l in pos_labels])
            neu_score = max([scores.get(l, 0) for l in neu_labels])
            
            # Determinar etiqueta principal
            if neg_score >= pos_score and neg_score >= neu_score:
                label = "NEG"
            elif pos_score >= neu_score:
                label = "POS"
            else:
                label = "NEU"
            
            # Calcular intensidad emocional (qué tan lejos del neutro)
            emotional_intensity = max(neg_score, pos_score)
            
            return PromptAnalysis(
                text=text,
                sentiment_label=label,
                negativity_score=neg_score,
                positivity_score=pos_score,
                emotional_intensity=emotional_intensity
            )
            
        except Exception as e:
            print(f"⚠️ Error en análisis de sentimiento: {e}")
            return PromptAnalysis(
                text=text,
                sentiment_label="NEU",
                negativity_score=0.0,
                positivity_score=0.0,
                emotional_intensity=0.0
            )


# Instancia global (singleton)
_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Obtiene la instancia global del analizador de sentimiento."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
