from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Extract data from the pdf file

DATA_PATH = "Data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)

#print('length of PDF pages',len(documents))

# create chunks
def create_chunks(extracted_data):
    text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunk = text_split.split_documents(extracted_data)
    return text_chunk

text_chunks = create_chunks(documents)
#print('Lenght of chunks',len(text_chunks))

# create vector embeddings
def get_embbedings():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embbedings()

# store embbeddings in FAISS 
DB_FAISS_PATH = 'vectorstore/db_faiss'
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)