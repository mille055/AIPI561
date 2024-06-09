import argparse
import hashlib
import os
import re
import json
#from docx import Document
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, PipelineTool
from openai import OpenAI
#import pinecone
from pinecone import init, Index, Pinecone, ServerlessSpec
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import logging
import fitz

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, openai_embedding_model='text-embedding-3-small', openai_engine='gpt-3.5-turbo', top_k=3, search_threshold=0.8, max_token_length=512, chunk_size=500, chunk_overlap=25, pinecone_index_name=None, llm_url=None, use_gpt=False, verbose=False):
        # pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        # index: can pass index or get environment variable; if none, use default
        if pinecone_index_name:
            self.pinecone_index_name = pinecone_index_name
        else:
            self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'duke-rads-chat')
        logger.info(f'The pinecone index is {self.pinecone_index_name}')
        # openai
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_embedding_model = openai_embedding_model
        self.openai_engine = openai_engine
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.use_gpt = use_gpt
        # text chunking and semantic search
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.search_threshold = search_threshold
        self.max_token_length = max_token_length
        self.prompt_instruction = "Please provide a concise summary to prospective students who are seeking answers to questions. Please generate text in complete sentences. Here is some context and the question: "
        self.verbose = verbose

        # our model
        if llm_url:
            self.llm_url = llm_url
        else:
            self.llm_url = os.getenv('LLM_URL', 'https://ej0lhmgikhbq6zp9.us-east-1.aws.endpoints.huggingface.cloud')
        self.llm_token = os.getenv('HUGGINGFACE_TOKEN')

        self.text_to_replace = ["© Copyright 2011-2024 Duke University", "Jump to navigation"]

        # Initialize Pinecone client
        #self.pc = pinecone.init(api_key=self.pinecone_api_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        if self.pinecone_index_name in self.pc.list_indexes().names():
            self.index = self.pc.Index(self.pinecone_index_name)
        else:
            self.create_pinecone()

    # Create the Pinecone store
    def create_pinecone(self, index_name = None):
        if index_name:
            self.pinecone_index_name = index_name
        
        if self.verbose:
            logger.info('Creating Pinecone index')
        if self.pinecone_index_name not in self.pc.list_indexes().names():
            logger.info(f'Creating index of name {self.pinecone_index_name}')
            self.pc.create_index(
                name=self.pinecone_index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
            if self.verbose:
                logger.info(f'Created new Pinecone index named {self.pinecone_index_name}')
        self.index = self.pc.Index(self.pinecone_index_name)

    # deletes a pinecone index
    def clear_pinecone(self, index_to_clear):
        logger.info(f'Clearing Pinecone index: {index_to_clear}')
        self.pc.delete_index(index_to_clear)

    # adds processed and embedded chunks from a json file to a pinecone index
    def populate_pinecone(self, json_file):
        logger.info(f'Populating Pinecone with data from {json_file}')
        self.load_and_process_json(json_file)

    def chunk_text(self, text):
        """
        Splits text into chunks with a specified maximum length and overlap,
        trying to split at sentence endings when possible.

        Input:
            self
            text (str): The input text.
        Output:
            chunks (list of str): Chunks of text.
        """
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = current_chunk[-self.chunk_overlap:] + ' ' + sentence
            else:
                current_chunk += sentence + ' '

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    def process_text(self, source, text, chunk_id):
        """
        Processes a grouping of the source, text, and chunk_id by getting embeddings
        and adding to the pinecone storage

        Input:
            self
            source (str): The url or path of the source of the chunk with the highest relevance
            text (str): The text of the chunk
            chunk_id (str): The number corresponding to the chunk
        Output:
            No output. Adjusts the pinecone vector storage
        """
        unique_id = hashlib.sha256(f"{source}_{chunk_id}".encode()).hexdigest()
        response = self.openai_client.embeddings.create(
            model=self.openai_embedding_model,
            input=[text]
        )
        embedding = response.data[0].embedding
        self.index.upsert(vectors=[{"id": unique_id, "values": embedding, "metadata": {"source": source, "text": text}}])

    def load_and_process_json(self, json_file):
        """
        Loads the json file and calls the text chunker

        Input:
            self
            json_file (str): The path/name of the json file to load and process
        Output:
            No output. Calls process_text to add to the vector store
        """
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for source, text in data.items():
            logger.info(f"Processing text: {source}")
            # cleaning text
            for phrase in self.text_to_replace:
                text = text.replace(phrase, " ")
                if self.verbose:
                    logger.debug(f'Replacing phrase: {phrase}')
            chunks = self.chunk_text(text)
            if isinstance(chunks, list):
                for i, chunk in enumerate(chunks):
                    self.process_text(source, chunk, i)
            else:
                logger.debug(chunks[0:40])
                self.process_text(source, chunks, 0)

    def semantic_search(self, query):
        """
        Performs a semantic search on the pinecone vector database

        Input:
            self
            query (str): The text input that is to be matched in the vector database

        Output:
            texts (list of str): the k highest matching chunks in a list
            sources (list of str): the source URLs of the k highest matching chunks
        """
        source_list = []
        texts = []
        try:
            response = self.openai_client.embeddings.create(
                model=self.openai_embedding_model,
                input=[query]
            )
            query_embedding = response.data[0].embedding

            results = self.index.query(vector=query_embedding, top_k=self.top_k, include_metadata=True)
            matches = [match for match in results["matches"]]
            for match in matches:
                source_list.append(match['metadata']['source'])
                texts.append(match['metadata']['text'])
            return texts, source_list
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return [], []

    def generate_response(self, query):
        """
        Generates the response based on the provided query by calling the semantic search,
        summarizing the top k chunks, and calling the function that sends the summarized chunks to
        either chatgpt or our fine-tuned model depending on self.use_gpt which can be toggled in the
        app.
        Inputs:
            self
            query (str): The user input question as a string.
        Outputs:
            response, sources (str, str): The output string from the chosen model and associated source
        """
        texts, sources = self.semantic_search(query)
        if texts:
            combined_chunks = " ".join(texts)
            summarized_response = self.openai_client.chat.completions.create(
                model=self.openai_engine,
                messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Summarize the following text:\n" + combined_chunks + "Please provide a concise summary to prospective students who are seeking answers to questions, starting directly with the key points without introductory phrases like 'The text discusses', 'The text outlines', 'The text covers', 'The text provided' or 'The text introduces'. Please generate text in complete sentences. Please also do not over-summarize, students need useful information"}],
                max_tokens=300,
                temperature=0.1
            )
            summarized_chunks = summarized_response.choices[0].message.content
            prompt = self.prompt_instruction + "\n Context: " + summarized_chunks + "\nUser Query:\n\n {} ###\n\n".format(query)
            response = self.integrate_llm(prompt)
            return response, sources
        else:
            return ("Sorry, I couldn't find a relevant response.", None)

    def integrate_llm(self, prompt):
        """
        Handles getting the response from the chosen model and the interface with the model for the
        prompt provided by generate_response

        Input:
            self
            prompt (str): The text input that forms the prompt as constructed in generate_response
        Output:
            chat_message (str): The output message from the model
        """
        if self.use_gpt:
            message = [{"role": "assistant", "content": "You are a trusted advisor helping to explain the text to prospective or current students who are seeking answers to questions"}, {"role": "user", "content": prompt}]
            if self.verbose:
                logger.debug(f'Debug: message is {message}')
                logger.debug(f'using gpt is: {self.use_gpt}')
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_engine,
                    messages=message,
                    max_tokens=300,
                    temperature=0.1
                )
                # Extracting the content from the response
                chat_message = response.choices[0].message.content

                return chat_message

            except Exception as e:
                logger.error(f"Error in generating response: {e}")
                return "Error in generating response"

        else:
            # Using the custom LLM HuggingFace endpoint
            headers = {
                "Authorization": f"Bearer {self.llm_token}"
            }
            payload = {
                "inputs": prompt
            }
            try:
                response = requests.post(self.llm_url, headers=headers, json=payload)
                response_data = response.json()
                response_text = response_data[0].get("generated_text", "No response generated.")
                response_text = response_text.replace(self.prompt_instruction, ' ').replace('#', ' ').split('User Query')[0]
                for phrase in self.text_to_replace:
                    response_text = response_text.replace(phrase, ' ')

                return response_text
            except Exception as e:
                logger.error(f"Error in connecting to the HuggingFace API: {e}")
                return "Error in connecting to the HuggingFace API. Please wait a few minutes or try using GPT (toggle above). Additionally, you can click the 'View Source' button to view a relevant web page."

    def remove_bullet_points(self, text):
        """
        Removes common bullet points and similar characters from the text.

        Input:
            self
            text (str): The input text.
        Output:
            text (str): The cleaned text without bullet points.
        """
        bullet_points = ['•', '–']
        for bullet in bullet_points:
            text = text.replace(bullet, '')
        # Remove multiple spaces caused by the replacements
        text = re.sub(r'\s+', ' ', text)
        return text

    def process_pdf(self, pdf_file):
        """
        Extracts text from a PDF file and processes it into chunks.

        Input:
            self
            pdf_file (str): The path to the PDF file.
        Output:
            No output. Calls process_text to add to the vector store.
        """
        doc = fitz.open(pdf_file)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            source = f"{os.path.basename(pdf_file)}_page_{page_num + 1}"
            text = self.clean_extracted_text(text)
            text = self.remove_bullet_points(text)
            chunks = self.chunk_text(text)
            if isinstance(chunks, list):
                for i, chunk in enumerate(chunks):
                    self.process_text(source, chunk, i)
            else:
                self.process_text(source, chunks, 0)

    def process_docx(self, docx_file):
        """
        Extracts text from a DOCX file and processes it into chunks.

        Input:
            self
            docx_file (str): The path to the DOCX file.
        Output:
            No output. Calls process_text to add to the vector store.
        """
        doc = Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        text = self.clean_extracted_text(text)
        text = self.remove_bullet_points(text)
        source = os.path.basename(docx_file)
        chunks = self.chunk_text(text)
        if isinstance(chunks, list):
            for i, chunk in enumerate(chunks):
                self.process_text(source, chunk, i)
        else:
            self.process_text(source, chunks, 0)

    def process_directory(self, directory):
        """
        Processes all PDF and DOCX files in the given directory.

        Input:
            self
            directory (str): The path to the directory containing PDF and DOCX files.
        Output:
            No output. Calls process_pdf or process_docx for each file found.
        """
        logger.info(f'Processing directory: {directory}')
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith('.pdf'):
                    logger.info(f'Processing PDF file: {file_path}')
                    self.process_pdf(file_path)
                elif file.lower().endswith('.docx'):
                    logger.info(f'Processing DOCX file: {file_path}')
                    self.process_docx(file_path)

    def process_url(self, url, auth=None):
        """
        Extracts text from a URL and processes it into chunks.

        Input:
            self
            url (str): The URL of the webpage to extract text from.
            auth (tuple, optional): A tuple containing username and password for authentication, if needed.
        Output:
            No output. Calls process_text to add to the vector store.
        """
        logger.info(f'Processing URL: {url}')
        response = requests.get(url, auth=auth)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')

        source = url
        chunks = self.chunk_text(text)
        if isinstance(chunks, list):
            for i, chunk in enumerate(chunks):
                self.process_text(source, chunk, i)
        else:
            self.process_text(source, chunks, 0)

    def process_urls(self, urls, auth=None):
        """
        Extracts text from multiple URLs and processes it into chunks.

        Input:
            self
            urls (list of str): The list of URLs to extract text from.
            auth (tuple, optional): A tuple containing username and password for authentication, if needed.
        Output:
            No output. Calls process_text to add to the vector store.
        """
        for url in urls:
            self.process_url(url, auth)
    
    def extract_text_with_pymupdf(self, pdf_file):
        doc = fitz.open(pdf_file)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def clean_extracted_text(self, text):
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Add space after punctuation if missing
        text = re.sub(r'(?<=[.,])(?=[^\s])', r' ', text)
        return text

    def extract_text_from_docx(self, docx_file):
        doc = Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text



# Example usage with command-line argument for specifying the JSON file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load text data from various sources and process with RAG.")
    parser.add_argument("--json_file", help="Path to the JSON file containing text data.")
    parser.add_argument("--pdf_file", help="Path to the PDF file containing text data.")
    parser.add_argument("--docx_file", help="Path to a DOCX file containing text data.")
    parser.add_argument("--directory", help="Path to the directory containing multiple PDF and DOCX files.")
    parser.add_argument("--url", help="URL of the webpage containing text data.")
    parser.add_argument("--urls", nargs='+', help="List of URLs of webpages containing text data.")
    parser.add_argument("--username", help="Username for authenticating URL access.")
    parser.add_argument("--password", help="Password for authenticating URL access.")
    parser.add_argument("--clear", help="Clear the pinecone index.")
    parser.add_argument("--create", help = "Create a new pinecone index with this name.")
    args = parser.parse_args()

    # Initialize your RAG instance
    rag = RAG(pinecone_index_name=args.create, verbose=True)

    if args.json_file:
        rag.populate_pinecone(args.json_file)

    if args.pdf_file:
        rag.process_pdf(args.pdf_file)

    if args.docx_file:
        rag.process_docx(args.docx_file)

    if args.directory:
        rag.process_directory(args.directory)

    if args.clear:
        rag.clear_pinecone(args.clear)

    if args.create:
        rag.create_pinecone(args.create)

    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)

    if args.url:
        rag.process_url(args.url, auth=auth)

    if args.urls:
        rag.process_urls(args.urls, auth=auth)

    
