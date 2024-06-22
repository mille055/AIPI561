from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz
import io
from PIL import Image
from rag import RAG
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# Initialize RAG
rag = RAG(use_gpt=True)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.json
    query = data['query']
    response_text, sources = rag.generate_response(query)
    return jsonify({'response': response_text, 'sources': sources})

@app.route('/get_page_image', methods=['GET'])
def get_page_image():
    pdf_file_path = request.args.get('pdf_file_path')
    page_num = int(request.args.get('page_num'))
    
    pdf_document = fitz.open(pdf_file_path)
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap()
    
    image_bytes = io.BytesIO(pix.tobytes(output="png"))
    image_bytes.seek(0)
    img = Image.open(image_bytes)
    
    return jsonify({'image': img.tobytes()})

if __name__ == '__main__':
    app.run(debug=True)
