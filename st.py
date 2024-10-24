from langchain_community.document_loaders import TextLoader
from PIL import Image
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("Employee Handbook copy.pdf")
docs  = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
                               chunk_size = 100,
                               chunk_overlap = 20, 
                               )

chunk_documents = text_splitter.split_documents(docs)

## Vector EMbeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunk_documents,
                          OpenAIEmbeddings()
                          )


## Design chat prompt template 

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer helpful.
< context>
{context} 
</ context>
Question : {input}                                         
"""                                   )

from langchain_community.llms import openai
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")  #gpt-3.5-turbo

## Chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = db.as_retriever()

## Retriver chain
from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever , document_chain)

# response = retrieval_chain.invoke({"input" : "Work hours of an employee"})
import streamlit as st
st.set_page_config(layout="wide")
######################## UI Design Start ####################################

############################ UI Design END #################################
# Load and display the hero image
hero_image_path = "/Users/sandeep/Documents/Lang_Chain/RAG_Chatbot/Photo.png"
hero_image = Image.open(hero_image_path)
# st.image(hero_image, use_column_width='always')

image_path = "/Users/sandeep/Documents/Lang_Chain/RAG_Chatbot/Wismettac.png"
logo = Image.open(image_path)

st.markdown(
    """
    <style>
    .overlay-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-size: 48px;  // Adjust based on your preference
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image(hero_image, use_column_width='always')

# Display the title in the second column
st.markdown(
    """
    <style>
    /* Importing Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Abril+Fatface&display=swap');

    /* Targeting the main title to center it and apply the font */
    .title {
        text-align: center;
        font-size: 80px; /* Adjust size as needed */
        font-weight: bold;
        font-family: 'Abril Fatface', serif; /* Using the imported Google Font */
        color: #400000; /* Optional: change text color */
    }

    /* Targeting the text input label for larger font size */
    .stTextInput label {
        font-size: 20px; /* Increase font size */
    }

    /* Styling the text input field itself */
    .stTextInput input {
        font-size: 20; /* Adjust this as you see fit */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Use the title with the specific class for styling
st.markdown('<p class="title"> Employee Assistant Bot</p>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* Targeting the text input label for larger font size */
    .stTextInput label {
        font-size: 24px; /* Increase font size of the label */
        color: #333; /* Optional: change label text color */
    }

    /* Styling the text input field itself for larger font size */
    .stTextInput input {
        font-size: 24px; /* Increase the font size of the input field */
        font-family: 'Abril Fatface', serif;
        height: 50px; /* Optional: increase the height of the input field */
    }

    /* Increasing the font size of the placeholder text within the input field */
    .stTextInput input::placeholder {
        font-family: 'Abril Fatface', serif;
        font-size: 24px; /* Setting the font size of the placeholder text */
        color: #888; /* Optional: changing the placeholder text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your text input field
user_query = st.text_input("Enter your question:", "")
st.markdown(
    """
    <style>
    .output-text {
        @import url('https://fonts.googleapis.com/css2?family=Abril+Fatface&display=swap');
        font-size: 24px; /* Larger font size for output text */
         font-family: 'Prociono', serif;
        color: #002D04; /* Optional: change text color */
        padding: 10px; /* Optional: add padding */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Now, use this class when displaying the response:
if user_query:
    try:
        response = retrieval_chain.invoke({"input": user_query})
        # Display the response using the new 'output-text' class
        st.markdown(f'<div class="output-text">{response["answer"]}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error("An error occurred: " + str(e))


## Sample Queries

# What are the working hours?
# What should I do if someone harasses me?
# What is the overtime pay?
# What is the company's attendance policy, and what procedures should I follow if I need to take more than 5 days off?   


# conda create -p venv python==3.10         
# conda activate venv/
# pip install -r re.txt
# conda activate venv/conda activate venv/