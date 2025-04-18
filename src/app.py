from time import sleep
from llama_index.core import Settings
from engine.ChatEngine import ChatEngine
from dotenv import load_dotenv
from vectorstore.ingestion import insert_into_chroma_db
from utils.file_reader import read_pdf_to_documents

load_dotenv()

chat = None

if __name__ == "__main__":
    llm = ChatEngine.get_llm()
    embed_model = ChatEngine.get_embedding_model()

    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = read_pdf_to_documents("assets/prompt_engineering.pdf")

    insert_into_chroma_db(documents)

    if chat is None:
        chat_engine = ChatEngine()

    while True:
        prompt = input("How can I help you? \nT")

        result = chat_engine.query(prompt)
        print(result)




