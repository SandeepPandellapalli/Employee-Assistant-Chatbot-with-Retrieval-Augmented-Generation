# Employee Assistant Chatbot with Retrieval Augmented Generation

This Streamlit app functions as an Employee Assistant Bot, designed to quickly retrieve and provide information from an Employee Handbook using natural language processing. It integrates LangChain for document retrieval and OpenAI's GPT models for processing and generating responses.

## Features

- **Document Loader**: Loads and processes PDF documents to be used as a knowledge base.
- **Text Splitter**: Splits large texts into manageable chunks to optimize the retrieval process.
- **Vector Embeddings**: Converts text chunks into vector embeddings for efficient similarity searches.
- **Chat Prompt Template**: Structures the chatbot responses to ensure relevance and clarity.
- **Chain Integration**: Combines document retrieval and chat response to generate coherent answers.
- **UI Design**: A user-friendly interface that allows employees to interact with the bot through natural language queries.

## Setup

To run this project locally, you will need Python 3.10 and the ability to install several dependencies.

### Environment Setup

Create a virtual environment and activate it:

```bash
conda create -p venv python=3.10
conda activate venv/

### Install Dependencies

Install the required Python libraries specified in `requirements.txt` (also referred to as `re.txt`):

```bash
pip install -r requirements.txt


