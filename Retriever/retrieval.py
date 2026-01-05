import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_astradb import AstraDBVectorStore
from typing import List, Tuple
from langchain_core.documents import Document
from dotenv import load_dotenv
from config.config_loader import load_config
from utils.model_loader import ModelLoader

class Retriever:
    """Utility class to handle retrieval operations from AstraDB Vector Store"""
    
    def __init__(self):
        print("Initializing Retriever...")
        self.config = load_config()
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None

    def _load_env_variables(self):
        """Load environment variables from .env file"""
        print("Loading environment variables for Retriever...")
        load_dotenv()
        required_vars = ["OPENAI_API_KEY", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_API_ENDPOINT"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

    def create_retriever(self):
        if self.vstore is None:

            """The data is already ingested in the AstraDB Vector Store.
            This function loads the retriever from the existing vector store.
            """
            print("Loading AstraDB Vector Store retriever...")
            # making a connection to the v ector store in AstraDB
            collection_name = self.config["astra_db"]["collection_name"]
            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),  #comes from class ModelLoader in utilis folder
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token
            )
        print("AstraDB Vector Store loaded successfully.")

        if self.retriever is None:
            top_k = self.config["retriever"]["top_k"]
            print(f"Setting up retriever with top_k={top_k}...")
            retriever = self.vstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            print("Retriever loaded successfully.")
            return retriever
        
    def call_retriever(self,query:str) ->List[Document]:
        """Call the retriever with a user query to fetch relevant documents."""
        retriever = self.create_retriever()
        print(f"Retrieving documents for query: {query}")
        output = retriever.invoke(query)
        return output

if __name__ == "__main__":
    retriever_obj = Retriever()
    user_query = "can you suggest good laptops with highest ratings and good reviews?"
    results = retriever_obj.call_retriever(user_query)
    for idx, doc in enumerate(results):
        print(f"Document {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")
    