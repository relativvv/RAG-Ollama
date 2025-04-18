from time import sleep
from typing import List
from dotenv import load_dotenv
from llama_index.core.indices.base import IndexType
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings,
)
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.extractors import KeywordExtractor, QuestionsAnsweredExtractor
from llama_index.core.schema import MetadataMode
from engine.ChatEngine import ChatEngine


def insert_into_chroma_db(documents: List[Document]) -> IndexType:
    load_dotenv()

    vector_store = ChatEngine.create_or_load_chroma_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llm = ChatEngine.get_llm()
    embed_model = ChatEngine.get_embedding_model()
    node_parser = SentenceSplitter.from_defaults()

    Settings.chunk_size = 512
    Settings.chunk_overlap = 30
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.node_parser = node_parser

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        transformations=[
            KeywordExtractor(keywords=3),
            QuestionsAnsweredExtractor(questions=3, metadata_mode=MetadataMode.EMBED)
        ],
        show_progress=True,
    )
    return index
