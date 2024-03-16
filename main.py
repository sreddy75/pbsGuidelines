from dotenv import load_dotenv
import os

import streamlit as st

from src.document_loader import DocumentLoader
from src.openAI_helper import OpenAIHelper
from src.prompt_generator import PromptGenerator

load_dotenv()

# Access environment variables
#openai_api_key = os.getenv("OPENAI_API_KEY")
doc_path = os.getenv("DOC_PATH")

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
                               key="openai_api_key_input"                               
                               )
    return input_text

openai_api_key = get_api_key()

st.markdown("## query the PBAC guideline doc")

def get_text():
    input_text = st.text_area(label="Query Input", label_visibility='collapsed', placeholder="Your query...", key="query_input")
    return input_text

query_input = get_text()

st.button("*ASK*", type='primary')

if query_input:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

llm = OpenAIHelper.loadllm(openai_api_key)
texts = DocumentLoader.load_documents_in_chunks(doc_path, 'txt', 1000, 0)
# embed texts and get retriever for the llm 
qa = OpenAIHelper.get_query_retriever_for_llm(openai_api_key, texts)
st.markdown("### Response:")

if len(query_input.split(" ")) <= 1:
    #do nothing prompt for a question
    st.write("Please enter a question.")
elif len(query_input.split(" ")) > 700:
    st.write("Please enter a shorter question. The maximum length is 700 words.")
    st.stop()
else:
    prompt_with_query = PromptGenerator.prompt.format(query=query_input)    
    # Run a query
    query = prompt_with_query
    response = qa({"query": query})
    st.write(response['result'])






