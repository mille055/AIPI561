import unittest
from rag import RAG
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logger.info(f'Testing setup of RAG class')
        cls.rag = RAG(verbose=True)

    def test_initialization(self):
        self.assertIsInstance(self.rag, RAG)
        logger.info(f'Testing initialization of RAG class')

    def test_create_pinecone(self):
        index_name = 'test-index'
        self.rag.create_pinecone(index_name=index_name)
        self.assertIn(index_name, self.rag.pc.list_indexes().names())
        self.rag.clear_pinecone(index_name)

    def test_clear_pinecone(self):
        index_name = 'test-index'
        self.rag.create_pinecone(index_name=index_name)
        self.rag.clear_pinecone(index_name)
        self.assertNotIn(index_name, self.rag.pc.list_indexes().names())

    def test_chunk_text(self):
        logger.info(f'Testing text chunking')
        text = "This is a sentence. This is another sentence. This is yet another sentence."
        chunks = self.rag.chunk_text(text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_remove_bullet_points(self):
        logger.info(f'Testing bullet point removal')
        text = "• Bullet point one\n– Bullet point two\n* Bullet point three"
        cleaned_text = self.rag.remove_bullet_points(text)
        self.assertNotIn('•', cleaned_text)
        self.assertNotIn('–', cleaned_text)
        self.assertNotIn('*', cleaned_text)

    def test_clean_extracted_text(self):
        logger.info(f'Testing clean extracted text.')
        text = "This is a sentence. This is another sentence.This is yet another sentence."
        cleaned_text = self.rag.clean_extracted_text(text)
        self.assertNotIn('  ', cleaned_text)
        self.assertIn('. ', cleaned_text)

    def test_process_text(self):
        text = "This is a sample text for embedding."
        source = "test_source"
        chunk_id = 0
        self.rag.process_text(source, text, chunk_id)
        response = self.rag.index.query(vector=self.rag.openai_client.embeddings.create(
            model=self.rag.openai_embedding_model, input=[text]).data[0].embedding, top_k=1, include_metadata=True)
        self.assertEqual(response['matches'][0]['metadata']['source'], source)

    # def test_process_docx(self):
    #     # Create a temporary DOCX file
    #     from docx import Document
    #     doc = Document()
    #     doc.add_paragraph("This is a test paragraph.")
    #     doc_path = "test.docx"
    #     doc.save(doc_path)

    #     self.rag.process_docx(doc_path)
    #     os.remove(doc_path)  # Clean up the temporary file

        # # Check if the document content was processed
        # response = self.rag.index.query(vector=self.rag.openai_client.embeddings.create(
        #     model=self.rag.openai_embedding_model, input=["This is a test paragraph."]).data[0].embedding, top_k=1, include_metadata=True)
        # self.assertEqual(response['matches'][0]['metadata']['source'], os.path.basename(doc_path))

    def test_process_pdf(self):
        # Create a temporary PDF file
        import fitz  # PyMuPDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is a test text.")
        pdf_path = "test.pdf"
        doc.save(pdf_path)

        self.rag.process_pdf(pdf_path)
        os.remove(pdf_path)  # Clean up the temporary file

        # Check if the PDF content was processed
        response = self.rag.index.query(vector=self.rag.openai_client.embeddings.create(
            model=self.rag.openai_embedding_model, input=["This is a test text."]).data[0].embedding, top_k=1, include_metadata=True)
        self.assertEqual(response['matches'][0]['metadata']['source'], f"{os.path.basename(pdf_path)}_page_1")

if __name__ == '__main__':
    unittest.main()
