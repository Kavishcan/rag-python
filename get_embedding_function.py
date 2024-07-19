from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import os
from langchain_openai import OpenAIEmbeddings


def get_embedding_function1():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def get_embedding_function():
    os.environ["OPENAI_API_KEY"] = "sk-proj-JuGeFyd7UOH9Ly37Qkm8T3BlbkFJjxQvOFOfj9aAOD5yEs0K"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return embeddings
