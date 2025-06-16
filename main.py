import os
from langchain.chat_models import init_chat_model
from read_pdf import extract_pdf_text
from store_text import create_vectorstore
from memory import get_conversational_chain
import streamlit as st


def main():
    st.title("📄 PDF Q&A Assistant 🤖")

    # Get OpenAI key securely only once
    if "OPENAI_API_KEY" not in os.environ:
        api_key = st.text_input("Enter your OpenAI API Key", type="password", key="api_input")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.warning("Please enter your OpenAI API key.")
            return

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        text = extract_pdf_text(uploaded_file)
        vs = create_vectorstore(text)
        qa_chain = get_conversational_chain(vs)

        question = st.text_input("Ask a question:", key="question_input")
        if question:
            response = qa_chain.run(question)
            st.write(response)


if __name__ == "__main__":
    main()