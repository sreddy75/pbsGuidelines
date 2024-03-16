from dotenv import load_dotenv
import magic
import os
import nltk
import sys

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
import magic
from dotenv import load_dotenv
import os
import nltk
import sys

load_dotenv()

# Access environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# nltk.download('averaged_perceptron_tagger')
# pip install unstructured
# Other dependencies to install https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
# pip install python-magic-bin
# pip install chromadb

loader = DirectoryLoader('/Users/sreddy/Downloads/Medicare/PBS_Guidelines', glob='**/*.txt')

# Load up your text into documents
documents = loader.load()

# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Split your documents into texts
texts = text_splitter.split_documents(documents)

# Turn your texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Get your docsearch ready
docsearch = FAISS.from_documents(texts, embeddings)

# Load up your LLM
llm = OpenAI(openai_api_key=openai_api_key)

# Create your Retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

# Run a query
query = "What treatment details are required for a listing"
result =  qa.run(query)

print(result)