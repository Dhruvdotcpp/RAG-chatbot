# Importing all the neccessary models for pdf processing(read, chunk, embed, and store)

# Importing PyPDFLoader to open and read pdf files
from langchain_community.document_loaders import PyPDFLoader

# Import text spliter to break long text into smaller overlapping chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import embeddings model to convert text into vectors
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Import ChromaDB vector store to store vectors and corresponding texts and to also perform searching operation
from langchain_community.vectorstores import Chroma



# Main Function (call this with any PDF path)
def ingestPdf(pdf_path: str):
    # pdf_path: str -> means this function expects a string like "docs/myfile.pdf"
    print(f"Starting ingestion for: {pdf_path}")

    # Stage-1 Loading
    # PyPDFLoader reads each page and returns list of document object
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # documents is a list like:
    # [Document(page_content="page 1 text", metadata={page: 0}),
    #  Document(page_content="page 2 text", metadata={page: 1}),
    #  ...]

    print(f"Loaded {len(documents)} pages from PDF")

    # Ṣtage-2 Chunking
    # RecursiveCharacterTextSplitter tries to split on paragraphs first, then sentences, then words — to keep meaning intact
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,   # Each chunk is 500 characters long
        chunk_overlap = 50  # Last 50 characters of a particular chunk are overlap with the first 50 character of the next chunk 
    )

    # split_documents takes our list of pages and splits each page into chunks
    chunks = splitter.split_documents(documents)

    print(f"✅ Split into {len(chunks)} chunks")