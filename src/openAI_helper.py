from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

class OpenAIHelper:

    def get_query_retriever_for_llm(api_key, texts):
        docsearch = OpenAIHelper.createEmbeddingsFromTexts(texts, api_key)
        llm = OpenAIHelper.loadllm(api_key)
        return OpenAIHelper.queryRetriever(llm, docsearch)
        

    def loadllm(api_key):
        return OpenAI(openai_api_key=api_key)
        
    def createEmbeddingsFromTexts(texts, api_key):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.from_documents(texts, embeddings)

    def queryRetriever(llm, docsearch):
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
