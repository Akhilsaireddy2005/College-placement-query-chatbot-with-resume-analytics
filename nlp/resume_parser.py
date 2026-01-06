"""
Resume Parser Module
Extracts text from PDF, DOC, DOCX, and plain text files
"""

import os
import PyPDF2
from docx import Document
from config import ALLOWED_EXTENSIONS


class ResumeParser:
    """Parse resumes from various file formats"""
    
    def __init__(self):
        self.supported_formats = ALLOWED_EXTENSIONS
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text
    
    def extract_text_from_txt(self, file_path):
        """Extract text from TXT file"""
        text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
        return text
    
    def parse(self, file_path):
        """
        Parse resume file and extract text
        
        Args:
            file_path: Path to the resume file
        
        Returns:
            Extracted text string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.rsplit('.', 1)[1].lower() if '.' in file_path else ''
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file_path)
        elif file_extension == 'doc':
            # Try to read as DOCX (may not work for old .doc files)
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

