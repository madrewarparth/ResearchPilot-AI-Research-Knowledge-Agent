"""
Configuration module for ResearchPilot.
Centralizes all application parameters, model names, and system paths.
Supports environment variable overrides via python-dotenv.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class Settings:
    # Application Info
    APP_NAME: str = "ResearchPilot"
    APP_TAGLINE: str = "Free, Local-First AI Research & Knowledge Agent"
    APP_VERSION: str = "1.0.0"

    # Ollama Service
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Vector Storage
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "researchpilot_collection")

    # Ingestion & Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    MIN_CHUNK_CHARS: int = 50

    # Retrieval & Evidence
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    
    # Offline & Fallback Mode
    # If Ollama is not active, ResearchPilot can fall back to local embedding/extractive synthesis
    # ensuring full testability and portfolio presentation without runtime crashes.
    ENABLE_OFFLINE_FALLBACK: bool = os.getenv("ENABLE_OFFLINE_FALLBACK", "true").lower() in ("true", "1", "yes")

    # Agent Settings
    MAX_RETRY_QUERY_REWRITE: int = 2
    SUFFICIENCY_THRESHOLD: float = 0.55

    @classmethod
    def check_ollama_connectivity(cls, host: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if Ollama service is reachable and list downloaded models.
        """
        import urllib.request
        import json
        
        target_host = host or cls.OLLAMA_HOST
        try:
            req = urllib.request.Request(f"{target_host.rstrip('/')}/api/tags", headers={"User-Agent": "ResearchPilot"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return {
                        "online": True,
                        "host": target_host,
                        "models": models,
                        "has_llm": any(cls.LLM_MODEL in m for m in models),
                        "has_embedding": any(cls.EMBEDDING_MODEL in m for m in models),
                        "message": f"Connected to Ollama at {target_host} ({len(models)} models available)."
                    }
        except Exception as e:
            return {
                "online": False,
                "host": target_host,
                "models": [],
                "has_llm": False,
                "has_embedding": False,
                "message": f"Ollama not reachable at {target_host} ({type(e).__name__}). Using offline fallback if enabled."
            }
        return {
            "online": False,
            "host": target_host,
            "models": [],
            "has_llm": False,
            "has_embedding": False,
            "message": f"Ollama service returned unexpected status."
        }

settings = Settings()
