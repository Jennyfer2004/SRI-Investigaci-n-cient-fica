from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

import requests
    
    
# ==== Definición del LLM personalizado ====
class HuggingFaceLLM(LLM):
    """Implementación personalizada de LLM que utiliza la API de Hugging Face.
    
    Args:
        api_url(str): URL del endpoint de la API de Hugging Face.
        api_token(str): Token de autenticación para la API.
        max_new_tokens(int): Número máximo de tokens a generar (default: 500).
        temperature(float): Parámetro de temperatura para la generación (default: 0.7).
    """
    api_url: str = Field(...)
    api_token: str = Field(...)
    max_new_tokens: int = Field(default=500)
    temperature: float = Field(default=0.7)

    def _call(self, prompt: str, stop = None):
        """
        Genera texto a partir de un prompt usando la API de Hugging Face.
        
        Args:
            prompt (str): Texto de entrada para la generación.
            stop(Optional[List[str]]): Lista de secuencias para detener la generación.    
        Returns:
            str: Texto generado por el modelo.
        """
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature
            }
        }
        r = requests.post(self.api_url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)

    @property
    def _llm_type(self) :
        return "huggingface_custom"

    @property
    def _identifying_params(self) -> dict:
        """Parámetros identificadores del modelo."""
        return {
            "api_url": self.api_url,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }

class Query(BaseModel):
    """Modelo de datos para las consultas de los usuarios
    Attributes:
        question(str): Pregunta del usuario.
        user_id(Optional[str]): Identificador del usuario (opcional).
    """
    question: str
    user_id: Optional[str] = None
    
class RecommendationRequest(BaseModel):
    user_id: str

class FeedbackRequest(BaseModel):
    """Modelo de datos para el feedback de los usuarios.
    Attributes:
        user_id(str): Identificador del usuario.
        question(str): Pregunta original.
        was_helpful (bool): Indica si la respuesta fue útil.
        feedback_text (Optional[str]): Comentario adicional del usuario (opcional).
    """
    user_id: str
    question: str
    was_helpful: bool
    feedback_text: Optional[str] = None
