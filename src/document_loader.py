from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    
    def load_documents(location, docType):
        if docType=='txt':
            # Load up your text into documents
            loader = DirectoryLoader(location, glob='**/*.txt')            
            documents = loader.load()
            return documents
        elif docType=='pdf':
            #loader = PyPDFLoader(absolute_path+"/sample_data/pbac-guidelines-version-5.pdf", extract_images=True)
            loader = PyPDFLoader(location, extract_images=True)
            documents = loader.load()
            return documents        
        
    def chunkDocText(documents, chunk_size, chunk_overlap):
        # Get your text splitter ready
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)                
        texts = text_splitter.split_documents(documents)
        return texts
    
    def load_documents_in_chunks(location, docType, chunk_size, chunk_overlap):
        docs = DocumentLoader.load_documents(location, docType)
        return DocumentLoader.chunkDocText(docs, chunk_size, chunk_overlap)