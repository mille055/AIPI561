import streamlit as st
from pinecone import init, Index, Pinecone, ServerlessSpec
from rag import RAG
import os, io
from dotenv import load_dotenv
import fitz
from PIL import Image

# Load environment variables
load_dotenv()
use_gpt = True
use_faq = False

# Initialize RAG with environment variables or directly with your keys
rag = RAG(use_gpt=use_gpt)

# Streamlit page configuration
st.set_page_config(page_title="Duke Radiology Resident Chatbot", layout="wide")

# style of the user and bot messages
def bot_message(message):
    st.markdown(f'<div class="bot-message" style="display: flex; padding: 5px;">'
                f'<div style="background-color: #012169; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-left:15px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)

def user_message(message):
    st.markdown(f'<div class="user-message" style="display: flex; padding: 5px;">'
                f'<div style="background-color: #00539B; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-left:15px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)


def get_page_image(pdf_file_path, page_num):
    """
    Extracts and returns a specific page image from a PDF file.

    Args:
        pdf_file_path (str): The path to the PDF file.
        page_num (int): The page number to extract the image for.

    Returns:
        BytesIO object containing the image.
    """
    pdf_document = fitz.open(pdf_file_path)
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap()

    #image_bytes = io.BytesIO(pix.getPNGData())
    image_bytes = io.BytesIO(pix.tobytes(output="png"))
    return image_bytes


def run_UI():
    # Apply custom CSS for styling message boxes
    st.markdown("""
        <style>
        .stApp { background-color: #fafafa; }
        .stheader { background-color: #0577B1}
        .user-message { 
            background-color: #00539B; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 15px; 
            float: left;
            clear: both;
        }
        .bot-message { 
            background-color: #012169; 
            padding: 10px; 
            border-radius: 15px; 
            margin: 10px;
            float: left;
            clear: both;
        }
        .stChatInputContainer > div {
                background-color: #E5E5E5;
                border-color: #012169;
                padding: 10px;
                border-radius: 15 px;
                margin: 10px;
                float: left;
                clear: both;
        }
        .stChatInputContainer input:focus {
            border-color: #012169;
        }
        .st-checkbox label span {
            background-color: #012169;
            border-color: #012169;
            color: #012169;
        }
        .st-checkbox span {
                color: black;
        }
        .custom-button {
            background-color: #3F7D7B;
            color: white;
            padding: 10px 24px;
            margin: 10px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            text-align: center;
            display: inline-block;
            text-decoration: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # avatars
    avatar_user = 'assets/Blue_question_mark_icon.png'
    avatar_assistant = 'assets/duke_d_2.png'
    
    st.image('assets/duke_chapel_blue.png', caption='Duke University')

    # Initialize or retrieve the conversation history from the session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'use_gpt' not in st.session_state:
        st.session_state.use_gpt = use_gpt

    # if 'use_faq' not in st.session_state:
    #     st.session_state.use_faq = True
    
     # Add a checkbox widget for toggling GPT functionality
    st.session_state.use_gpt = st.checkbox('Use GPT')
    rag.use_gpt = st.session_state.use_gpt

    # Add a checkbox widget for toggling faq functionality - for now, keeping always on
    #st.session_state.use_faq = st.checkbox('Use FAQs', value=st.session_state.use_faq)

    for message in st.session_state.conversation_history:
        # display the chat history
        with st.chat_message(message["role"], avatar = message["avatar"]):
            #st.markdown(message["content"])
            if message["role"]=="user":
                user_message(message["content"])
            else:
                bot_message(message["content"])

    # get new input and process
    if prompt := st.chat_input("What questions can I help you with?"):
        st.session_state.conversation_history.append({"role": "user", "content": prompt, "avatar": avatar_user})
        with st.chat_message("user", avatar = avatar_user):
            #display the user question 
            user_message(prompt)
        
        # get and display the response
        with st.chat_message("assistant", avatar=avatar_assistant):
            #whole_prompt = 'Please answer the following query and generate a response. please do not include phrases like the text discusses... or the text outlines...:' + prompt + 'and the following context may be helpful' + " ".join([message['content'] for message in st.session_state.conversation_history])

            #faq attempt
            # if st.session_state.use_faq:
            #     response_text, score = rag.get_similar_faq(prompt)
            #     if response_text:
            #         sources = ['https://mille055.github.io/duke_chatbot/data/faqs.html'] # static faq website
            #         response_text = 'From the FAQs:  ' + response_text + ' For more information from the FAQs click the "View Source" button below.'
                    
           
            # select the model to be used
            rag.use_gpt = st.session_state.use_gpt
            response_text, sources = rag.generate_response(prompt)

            # Append response to conversation history and display 
            st.session_state.conversation_history.append({"role": "assistant", "content": response_text, "avatar": avatar_assistant})        
            bot_message(response_text)
        
            # Store sources in session state to access them later
            st.session_state.sources = sources

        # # button to display source
        # if sources:
        #     source = sources[0]
        #     if st.button("View Source"):
        #         if 'http://' in source or 'https://' in source:
        #             st.markdown(f"<div style='text-align: right;'><a href='{source}' target='_blank'><button style='background-color: #3F7D7B; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;'>View Source</button></a></div>", unsafe_allow_html=True)
        #         elif 'pdf' in source:
        #             st.write('PDF source detected.')
        #             # Extract the filename and page number
        #             try:
        #                 pagenumber = int(source.split('_')[-1])
        #                 filename = os.path.join('data/docs/', source.split('_page')[0])
        #                 st.write('Page:', pagenumber)
        #                 st.write('Document:', filename)
        #                 # Get and display the PDF page image
        #                 image_bytes = get_page_image(filename, pagenumber - 1)
        #                 image = Image.open(image_bytes)
        #                 st.image(image, caption=f"Page {pagenumber} of {os.path.basename(filename)}")
        #             except Exception as e:
        #                 st.write(f"Error processing PDF: {e}")

        # button to display source
    if 'sources' in st.session_state:
        source = st.session_state.sources[0]
        #st.write(f"Source detected: {source}")  # Debug output to show the detected source

        if 'http://' in source or 'https://' in source:
            if st.button("View Source"):
                st.markdown(f"<div style='text-align: right;'><a href='{source}' target='_blank'><button style='background-color: #3F7D7B; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;'>View Source</button></a></div>", unsafe_allow_html=True)
        elif 'pdf' in source:
            #st.write('PDF source detected.')
            if st.button("View Source"):
                #st.markdown(f"<div style='text-align: right;'><button style='background-color: #3F7D7B; color: white; padding: 10px 24px; margin: 10px; border: none; border-radius: 12px; cursor: pointer;' onClick='window.location.href = window.location.href;'>View PDF Source</button></div>", unsafe_allow_html=True)
                try:
                    pagenumber = int(source.split('_')[-1])
                    filename = os.path.join('data/docs/', source.split('_page')[0])
                    #st.write(filename, pagenumber)
                    # Get and display the PDF page image
                    image_bytes = get_page_image(filename, pagenumber - 1)
                    image = Image.open(image_bytes)
                    st.image(image, caption=f"Page {pagenumber} of {os.path.basename(filename)}")
                except Exception as e:
                    st.write(f"Error processing PDF: {e}")

           
if __name__ == "__main__":
    run_UI()
