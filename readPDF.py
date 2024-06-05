from PyPDF2 import PdfReader


def read_pdf_content(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PdfReader(file)

        # Iterate through each page and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text


def read_text_file(file_path):
    text = ""
    with open(file_path, "r") as file:
        # Read the entire content of the file
        text = file.read()
    return text
