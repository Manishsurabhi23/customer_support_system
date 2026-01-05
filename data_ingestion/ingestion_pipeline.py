import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
from utils.model_loader import ModelLoader
from config.config_loader import load_config
import pandas as pd

class DataIngestion:
    """utility class to load embedding models and LLM's"""
    
    def __init__(self):
        print("Initializing Data Ingestion Pipeline...")
        self.config = load_config()
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()

    def _load_env_variables(self):
        """Load environment variables from .env file"""
        print("Loading environment variables...")
        load_dotenv()
        required_vars = ["OPENAI_API_KEY", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_API_ENDPOINT"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    
    def _get_csv_path(self) -> str:
        """Get the CSV file path from config"""
        print("Getting CSV file path...")
        current_dir = os.getcwd()
        #csv_path = os.path.join(current_dir,'data','flipkark_product_review.csv')
        csv_path = os.path.join(current_dir,'data',self.config["data_ingestion"]["file_name"])
        #csv_path = self.config["data_ingestion"]["csv_path"]
        return csv_path
    
    def _load_csv(self):
        """Load the product data from CSV and validate columns."""
        print(f"Loading CSV data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        expected_columns = {'product_title', 'rating', 'summary', 'review'}
        missing = expected_columns - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    
    def transform_data(self):
        """Transform raw CSV data into Document objects with metadata"""
        print("Starting data transformation...")
        #required_columns = self.product_data.columns#drop product id and any other unwanted columns
        # print(f"required columns are : {required_columns}")
        #required_columns = list(required_columns[1:])
        #print(required_columns)
        product_list = []
        for index,row in self.product_data.iterrows():
            object = {
                "product_name": row['product_title'],
                "product_rating": row['rating'],
                "product_summary": row['summary'],
                "product_review": row['review']
            }
            product_list.append(object)
        # print(f"total products list : {(product_list)}")
        documents = []
        for entry in product_list:
            metadata = {
                "product_name": entry['product_name'],
                "product_rating": entry['product_rating'],
                "product_summary": entry['product_summary']
            }
        #create document object
            doc = Document(page_content=entry['product_review'], metadata=metadata)
            documents.append(doc)
        print(f"transformed {len(documents)} documents.")
        return documents
    
    def store_in_vector_db(self, documents: List[Document]) -> Tuple[AstraDBVectorStore, List[str]]:
        """Store Document objects into AstraDB vector store"""
        collection_name = self.config["astra_db"]["collection_name"]
        vstore = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),  #comes from class ModelLoader in utilis folder
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token)
        inserted_ids = vstore.add_documents(documents)
        print(f"successfully inseterd {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids

    def run_pipeline(self):
        """
        Run the full data ingestion pipeline: transform the data and store into vectorDB

        """
        print("Running data ingestion pipeline...")
        documents = self.transform_data()
        vstore, inserted_ids = self.store_in_vector_db(documents)
        #optionally do a quick search
        query = "can you tell me the low budget headphone?"
        results = vstore.similarity_search(query)

        print(f"Sample search results for query: {query}")
        for res in results:
            print(f"\n content: {res.page_content}\n Metadata: {res.metadata}\n")

if __name__ == "__main__" :

    ingestion = DataIngestion()
    ingestion.run_pipeline()