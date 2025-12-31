import pandas as pd
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

class data_converter:
    def __init__(self):
        print("data convert has initialized")
        self.product_data = pd.read_csv(r'D:\krish_naik_data_science_course\NLP\customer_support_system\data\flipkart_product_review.csv') #need to use os library to read file path
       # print(self.product_data.head())

    def data_transformation(self):
        required_columns = self.product_data.columns#drop product id and any other unwanted columns
        # print(f"required columns are : {required_columns}")
        required_columns = list(required_columns[1:])
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
        docs = []
        for entry in product_list:
            metadata = {
                "product_name": entry['product_name'],
                "product_rating": entry['product_rating'],
                "product_summary": entry['product_summary']
            }
        #create document object
            doc = Document(page_content=entry['product_review'], metadata=metadata)
            docs.append(doc)
        print(f"sample document : {docs[0:4]}")
        return docs
if __name__ == "__main__":
    data_con = data_converter()
    data_con.data_transformation()
