# 🤖 AURA Chatbot Service

> **Microservicio de IA Conversacional para Apoyo Psicoemocional**  
> **Tecnologías:** FastAPI, Google Gemini AI, Transformers (RoBERTa), Python 3.11+

Servicio de chatbot que proporciona respuestas empáticas y contextualizadas basadas en el perfil psicoemocional del usuario, integrando análisis de sentimiento NLP y datos del sistema de clustering.

---

## 🚀 Quick Start

```bash
# 1. Crear entorno virtual
cd chatbot-service-aura
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar GEMINI_API_KEY

# 4. Iniciar el servicio
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

### Obtener API Key de Gemini

1. Ir a [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Iniciar sesión con cuenta de Google
3. Crear nueva API Key
4. Copiar y pegar en `.env`

---

## 📚 Documentación

- **Swagger UI:** http://localhost:8002/docs
- **ReDoc:** http://localhost:8002/redoc

---

## 🔗 Endpoints

| Método | Endpoint | Descripción |
|:-------|:---------|:------------|
| POST | `/api/v1/chat/message` | Enviar mensaje y recibir respuesta |
| GET | `/api/v1/chat/health` | Estado del servicio |
| GET | `/api/v1/chat/greeting` | Saludo inicial personalizado |

### Ejemplo de Request

```bash
curl -X POST http://localhost:8002/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Me siento muy solo últimamente"
  }'
```

### Ejemplo de Response

```json
{
  "message": "Entiendo cómo te sientes. Es muy válido sentirse solo a veces...",
  "metadata": {
    "intent_detected": "support",
    "risk_level": "MODERADO",
    "sentiment_label": "NEG",
    "negativity_score": 0.65,
    "requires_follow_up": true,
    "crisis_resources_included": false
  },
  "timestamp": "2025-12-09T15:30:00Z"
}
```

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    CHATBOT SERVICE                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   FastAPI   │──│  NLP Module │──│  Clustering Client  │ │
│  │   Routes    │  │  (RoBERTa)  │  │  (HTTP Async)       │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┴─────────────────────┘            │
│                          │                                  │
│              ┌───────────┴───────────┐                     │
│              │    Context Builder    │                     │
│              └───────────┬───────────┘                     │
│                          │                                  │
│              ┌───────────┴───────────┐                     │
│              │    Gemini Client      │                     │
│              └───────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   Google Gemini API   │
               └───────────────────────┘
```

---

## 🧠 Pipeline de Procesamiento

1. **Análisis de Sentimiento (RoBERTa)**
   - Detecta tono emocional del mensaje
   - Calcula índice de negatividad
   - Identifica intensidad emocional

2. **Clasificación de Intención**
   - Crisis (requiere intervención urgente)
   - Apoyo emocional (busca soporte)
   - Información (preguntas generales)
   - Saludo (conversación casual)

3. **Contexto de Clustering**
   - Consulta perfil de riesgo del usuario
   - Obtiene factores de riesgo históricos
   - Combina con análisis actual

4. **Generación de Respuesta (Gemini)**
   - System prompt con directrices psicoemocionales
   - Contexto del usuario inyectado
   - Respuesta empática y apropiada

---

## ⚠️ Manejo de Crisis

Cuando se detecta una situación de crisis, el sistema:

1. **Bypassa** la generación de Gemini
2. **Retorna** respuesta predefinida con recursos
3. **Incluye** líneas de ayuda profesional
4. **Marca** `crisis_resources_included: true`

### Recursos de Crisis (México)

- **Línea de la Vida:** 800-911-2000 (24 horas)
- **SAPTEL:** 55 5259-8121
- **Consejo Ciudadano:** 55 5533-5533

---

## 🔧 Configuración

### Variables de Entorno

| Variable | Descripción | Default |
|:---------|:------------|:--------|
| `GEMINI_API_KEY` | API Key de Google AI | **Requerido** |
| `GEMINI_MODEL` | Modelo de Gemini | `gemini-1.5-flash` |
| `GEMINI_TEMPERATURE` | Creatividad (0-1) | `0.7` |
| `CLUSTERING_SERVICE_URL` | URL del Clustering Service | `http://localhost:8001` |
| `SERVICE_PORT` | Puerto del servicio | `8002` |

---

## 📁 Estructura del Proyecto

```
chatbot-service-aura/
├── requirements.txt
├── .env.example
├── README.md
└── app/
    ├── __init__.py
    ├── main.py              # Entry point FastAPI
    ├── config.py            # Configuración
    ├── api/
    │   └── chat_routes.py   # Endpoints
    ├── models/
    │   └── schemas.py       # Pydantic schemas
    ├── nlp/
    │   ├── sentiment_analyzer.py  # RoBERTa
    │   └── prompt_classifier.py   # Intent detection
    └── services/
        ├── clustering_client.py   # HTTP client
        ├── user_context.py        # Context builder
        └── gemini_client.py       # Gemini API
```

---

## 🔗 Dependencias de Servicios

| Servicio | Puerto | Requerido | Descripción |
|:---------|:------:|:---------:|:------------|
| Clustering Service | 8001 | Opcional | Perfil de riesgo del usuario |
| Gemini API | - | **Sí** | Generación de respuestas |

> El servicio funciona sin Clustering Service, pero las respuestas no estarán contextualizadas con el perfil histórico del usuario.

---

## 📝 Consideraciones

1. **Este chatbot NO reemplaza atención profesional**
2. **Nunca proporciona diagnósticos médicos**
3. **Siempre deriva a profesionales en casos serios**
4. **Los datos de usuario deben manejarse con privacidad**

---

*Microservicio desarrollado para AURA - Sistema de Apoyo Psicoemocional para Jóvenes*
