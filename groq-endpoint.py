import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

GROQ_API_KEY = "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"
OPENAI_API_KEY = "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"

llm = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=GROQ_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)   

vector_store = Chroma(
    collection_name="langchain_chroma_db",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# Langsmith 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
# Groq:  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