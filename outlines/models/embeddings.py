"""Router for embedding models."""
from .hf_transformers import HuggingFaceEmbeddings
from .openai import OpenAIEmbeddings

openai = OpenAIEmbeddings
huggingface = HuggingFaceEmbeddings
