import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, level.upper())
    logging.getLogger().setLevel(numeric_level)


def save_results(result: Dict[str, Any], output_path: str) -> None:
    """Save RAG results to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def format_response(question: str, answer: str) -> Dict[str, str]:
    """Format Q&A response."""
    return {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }


def sanitize_metadata(metadata: dict) -> dict:
    """
    Sanitize metadata to ensure compatibility with ChromaDB:
    - Convert None to empty string
    - Convert lists to comma-separated strings
    - Convert all values to supported types (str, int, float, bool)
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, list):
            sanitized[key] = ",".join(str(item) for item in value)
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized
