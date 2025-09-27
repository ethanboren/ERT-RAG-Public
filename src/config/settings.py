import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Validate environment
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

# Define base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
VECTORSTORE_PATH = str(PROJECT_ROOT / "data" / "chroma")

# Model settings
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gpt-4o-mini"

# Initialize components
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
llm = init_chat_model(LLM_MODEL, model_provider="openai")
