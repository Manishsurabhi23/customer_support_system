import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config.config_loader import load_config
class ModelLoader:
    """utility class to load embedding models and LLM's"""
    def __init__(self):
        print("Model Loader Initialized")
        load_dotenv()
        self.config = load_config()
        self._validate_config()
    
    def _validate_config(self):
        """validate necessary env variables are set"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    def load_embeddings(self):
        """Load and return the embedding model based on config"""
        embedding_model_name = self.config["embedding_model"]["model_name"]
        embeddings = OpenAIEmbeddings(model=embedding_model_name)
        return embeddings
    
    def load_llm(self):
        """Load and return the LLM model based on config"""
        llm_model_name = self.config["llm"]["model_name"]
        llm = ChatOpenAI(model=llm_model_name)
        return llm