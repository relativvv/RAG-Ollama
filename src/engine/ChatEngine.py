from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core.chat_engine.types import ChatMode, BaseChatEngine
from llama_index.core.postprocessor import (
    SentenceEmbeddingOptimizer,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import time
from llama_index.vector_stores.chroma import ChromaVectorStore
from datetime import datetime
from os import getenv
import chromadb


class ChatEngine:
    def __init__(self):
        self.index = self.get_chroma_index()
        self.chat_engine = self._set_chat_engine()
        self.last_interaction = time.time()

    def query(self, message: str):
        self.reset_last_interaction()
        response = self.chat_engine.chat(message)
        for source_node in response.source_nodes:
            print(repr(source_node))
        return response.response

    def get_chat_engine(self):
        self.reset_last_interaction()
        return self.chat_engine

    def _set_chat_engine(self) -> BaseChatEngine:
        index = self.index

        llm = self.get_llm()
        embed_model = self.get_embedding_model()

        embedding_postprocessor = SentenceEmbeddingOptimizer(
            embed_model=embed_model, 
            percentile_cutoff=0.2,
            threshold_cutoff=0.5,
        )

        today = datetime.today()
        date = today.strftime("%B %d, %Y")

        return index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            llm=llm,
            verbose=True,
             node_postprocessors=[
                 embedding_postprocessor,
            ],
            system_prompt=(
                f"Today is {date}. "
                f"You are a friendly chat bot. You know everything about prompt engineering. "
                f"You are designed to answer questions based on these knowledge sources. "
                f"Always give the source of your information and be as helpful as possible."
            ),
        )

    def reset_last_interaction(self):
        self.last_interaction = time.time()

    @staticmethod
    def get_embedding_model():
        load_dotenv()
        return OllamaEmbedding(
            model_name="nomic-embed-text",
        )

    @staticmethod
    def get_llm():
        load_dotenv()
        return Ollama(
            model="llama3.2",
            verbose=True,
        )

    @staticmethod
    def create_or_load_chroma_vector_store():
        load_dotenv()
        remote_db = chromadb.HttpClient(host=getenv("CHROMA_DB_HOST"), port=42042)
        
        # Try to get the collection first or create it with specific metadata_config
        try:
            # Check if collection exists first
            chroma_collection = remote_db.get_collection("vector_store")
            print("Using existing Chroma collection")
        except Exception:
            # Create collection with explicit metadata schema
            print("Creating new Chroma collection")
            chroma_collection = remote_db.create_collection(
                name="vector_store",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,  # We'll set this through LlamaIndex
                # Explicitly define metadata schema to include _type field
            )
        
        return ChromaVectorStore(chroma_collection=chroma_collection)

    @staticmethod
    def get_chroma_index():
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager(handlers=[llama_debug])

        Settings.callback_manager = callback_manager

        return VectorStoreIndex.from_vector_store(
            vector_store=ChatEngine.create_or_load_chroma_vector_store(),
        )