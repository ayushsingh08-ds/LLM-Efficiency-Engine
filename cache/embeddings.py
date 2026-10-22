from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embedding_model: Optional[Any] = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Load and return a singleton sentence-transformer model instance."""
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(model_name)

    return _embedding_model


def prompt_to_vector(prompt: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
	"""Convert a prompt string into a dense embedding vector."""
	if not isinstance(prompt, str) or not prompt.strip():
		raise ValueError("prompt must be a non-empty string")

	model = get_embedding_model(model_name=model_name)
	vector = model.encode(prompt, normalize_embeddings=True)

	return vector.tolist()

