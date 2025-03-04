from src.helper import pdfLoader, splitText, download_hf_embeddings, text_chunks, embeddings
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# create db on pinecone, only run once
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'medibot'

pc.create_index(
    name=index_name,
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# embed each chunk and upsert the embeddings into your pinecone index
from langchain.vectorstores import Pinecone

docsearch = Pinecone.from_documents(
    documents = text_chunks,
    index_name=index_name,
    embedding=embeddings
)
