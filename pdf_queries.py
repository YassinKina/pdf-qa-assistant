from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def get_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
