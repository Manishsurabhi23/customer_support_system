# from langchain_astradb import AstraDBVectorStore
# from dotenv import load_dotenv
# import os
# import pandas as pd
# from data_ingestion.data_transform import data_converter  #folder.file import class
# from langchain_google_genai import GoogleGenerativeAIEmbeddings


# load_dotenv()

# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["ASTERA_DB_API_ENDPOINT"] = os.getenv("ASTRA_DB_API_ENDPOINT")
# os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# os.environ["ASTRA_DB_KEYSPACE"] = os.getenv("ASTRA_DB_KEYSPACE")


# class ingest_data:
#     def __init__(self):
#         print("data ingestion class has initialized")
#         self.embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
#         self.data_converter = data_converter()

#     def data_ingestion(self,status):
#         vstore = AstraDBVectorStore(
#             embedding=self.embeddings,
#             collection_name="chatbot_ecomm",
#             token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
#             api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
#         )
#         storage = status
#         if storage ==None:
#             docs = self.data_converter.data_transformation()
#             inserted_ids = vstore.add_documents(docs)
#             print(f"data ingestion is completed and inserted ids are : {inserted_ids}")
#         else:
#             return vstore #returning vstore if data is already ingested 
#         return vstore,inserted_ids#returning vstore and inserted ids
    



# if __name__ == "__main__":
#     data_ingest= ingest_data()
#     vstore,inserted_ids=data_ingest.data_ingestion()
#     print(f"\nInserted {len(inserted_ids)} documents into AstraDB Vector Store.")
#     result=vstore.similarity_search("can you tell me the low budget headphones?")
#     for res in result:
#         print(f"\nResult Document: {res.page_content}\nMetadata: {res.metadata}\n")

from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
from data_ingestion.data_transform import data_converter
from langchain_openai import OpenAIEmbeddings
# Load environment variables from .env
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
class ingest_data:
    def __init__(self):
        print("Data ingestion class initialized")

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
            )       

        self.data_converter = data_converter()

    def data_ingestion(self, status=None):
        """
        If status is None  -> ingest data into AstraDB
        If status is not None -> skip ingestion and reuse existing data
        """

        # Create / connect to existing vector store
        vstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name="chatbot_ecomm",
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )

        # Ingest data ONLY once
        if status is None:
            print("Status is None → ingesting data into vector DB...")

            docs = self.data_converter.data_transformation()
            inserted_ids = vstore.add_documents(docs)

            print(f"Ingestion completed. Inserted {len(inserted_ids)} documents.")
            return vstore, inserted_ids

        # Skip ingestion
        print("Status is NOT None → skipping data ingestion.")
        return vstore, []


if __name__ == "__main__":
    data_ingest = ingest_data()

    # 👇 First run: use status=None
    # 👇 Next runs: change status to any non-None value
    vstore, inserted_ids = data_ingest.data_ingestion(status=None)

    print(f"\nInserted {len(inserted_ids)} documents into AstraDB Vector Store.")

    # Test similarity search
    results = vstore.similarity_search(
        "can you tell me the low budget headphones?",
        k=3
    )

    for res in results:
        print(
            f"\nResult Document:\n{res.page_content}\nMetadata: {res.metadata}\n"
        )
