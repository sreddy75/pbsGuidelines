from dotenv import load_dotenv
import os
import sys

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

from src.document_loader import DocumentLoader
from src.openAI_helper import OpenAIHelper

load_dotenv()

# Access environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
doc_path = os.getenv("DOC_PATH")
# load and Split your documents into texts
print('***Doc_PATH', doc_path)
texts = DocumentLoader.load_documents_in_chunks(doc_path, 'txt', 1000, 0)
# embed texts and get retriever for the llm 
qa = OpenAIHelper.get_query_retriever_for_llm(openai_api_key, texts)
# Run a query
query = "What treatment details are required for a listing"
result =  qa.run(query)
print(result)