from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Access environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
doc_path = os.getenv("DOC_PATH")

print("key: ", openai_api_key)
print("path: ", doc_path)

st.set_page_config(page_title="Read PBS Guideline", page_icon=":robot:")

st.header("Parse PBS guidelines")

col1, col2 = st.columns(2)

with col1:
    st.markdown("The PBAC (Pharmaceutical Benefits Advisory Committee) Guidelines provide detailed instructions on \
                how to prepare a submission to list a new medicine or medicinal product on the Pharmaceutical Benefits Schedule for \
                public funding. This tool is powered by openAI llm and custom code to parse the guideline pdf directly and answer \
                and questions from the guide line in natural language made by suren Tippireddy as a POC to consume the PBAC guidelines in NLP \
                \n\n View Source Code on [Github](https://github.com/sreddy75")

with col2:
    st.image(image='pbs.jpeg', width=500, caption='https://pbac.pbs.gov.au/')

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  
                               placeholder="Ex: sk-2twmA8tfCb8un4...", 
                               key="openai_api_key_input",
                               value=openai_api_key                               
                               )
    return input_text

openai_api_key = get_api_key()

st.markdown("## PBAC guideline")

def get_text():
    input_text = st.text_area(label="Query Input", label_visibility='collapsed', placeholder="Your query...", key="query_input")
    return input_text

query_input = get_text()

st.button("*ASK*", type='primary')

if query_input:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

loader = DirectoryLoader(doc_path+'/', glob='**/*.txt')
#Load up your text into documents
documents = loader.load()

#loader = PyPDFLoader(doc_path+"pbac-v5.pdf", extract_images=True)
#documents = loader.load()

print("length of documents: ", len(documents))

# Get your text splitter ready
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Split your documents into texts
texts = text_splitter.split_documents(documents)
print("length of document splits: ", len(texts))
print("length of document split text[0]: ", texts[0])
# Turn your texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Get your docsearch ready
docsearch = FAISS.from_documents(texts, embeddings)
# Load up your LLM
llm = OpenAI(openai_api_key=openai_api_key)
# Create your Retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
        
st.markdown("### Response:")

template = """
    you are an expert research assistant around the pbac guidelines. your goal is to search the document and find 
    the information for the given query and provide an detailed explanation.

    also list the relevant sections of the document that contains the response as a bullet points at the end of the response.

    QUERY: {query}
    YOUR RESPONSE:
"""
prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

if len(query_input.split(" ")) <= 1:
    #do nothing prompt for a question
    st.write("Please enter a question.")
elif len(query_input.split(" ")) > 700:
    st.write("Please enter a shorter question. The maximum length is 700 words.")
    st.stop()
else:
    prompt_with_query = prompt.format(query=query_input)    
    # Run a query
    query = prompt_with_query
    response = qa({"query": query})
    st.write(response['result'])






