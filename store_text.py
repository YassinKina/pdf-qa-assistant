from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

def create_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore
