from PyPDF2 import PdfReader

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])