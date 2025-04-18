from llama_index.core import Document
import os
from pypdf import PdfReader


def read_pdf_to_documents(pdf_path: str, label: str = "pdf") -> list[Document]:
    documents = []
    
    try:
        reader = PdfReader(pdf_path)
        filename = os.path.basename(pdf_path)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip(): 
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "filename": filename,
                            "label": label,
                            "page_number": i + 1,
                            "total_pages": len(reader.pages),
                            "source": pdf_path,
                        },
                    )
                )
        
        print(f"Successfully processed PDF: {filename} - {len(documents)} pages extracted")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
    
    return documents
