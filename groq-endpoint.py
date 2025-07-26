import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

GROQ_API_KEY = ""
OPENAI_API_KEY = """

llm = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=GROQ_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

vector_store = Chroma(
    collection_name="langchain_chroma_db",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

