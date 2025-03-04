
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def pdfLoader(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents


extracted_data = pdfLoader("Data/")



# Split the text into chunks
def splitText(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(data)

text_chunks = splitText(extracted_data)


from langchain.embeddings import HuggingFaceEmbeddings

def download_hf_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2") # a 384 dimensional embedding
    return embeddings

embeddings = download_hf_embeddings()


