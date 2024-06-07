# Omaxe Chatbot Project

## Overview

This project aims to create an intelligent chatbot for Omaxe Pvt Ltd, a leading real estate company. The chatbot is designed to assist users by providing accurate and relevant information about Omaxe's properties, services, and other inquiries. It leverages advanced natural language processing (NLP) models to deliver smart, context-aware responses.

## Features

- Answers queries about Omaxe's commercial and residential properties.
- Provides information about the company's owner, Rohtas Goel.
- Handles variations in question phrasing to retrieve accurate answers.
- Offers a user-friendly interface for engaging with the chatbot.
- Includes a suggestion feature for commonly asked questions.
- Sends email notifications for new queries.
- Utilizes a chat memory buffer to maintain context in conversations.

## Technology Stack

- Flask: Web framework for building the chatbot application.
- HuggingFace: Provides pre-trained NLP models for generating responses.
- Llama Index: Used for creating an index of documents and efficient retrieval.
- HuggingFace Embeddings: For embedding text data.
- Python: Core programming language for the application.
- HTML/CSS/JavaScript: Frontend technologies for the web interface.

## Setup Instructions

Follow these steps to set up and run the Omaxe Chatbot locally.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- HuggingFace account with an API token

### Clone the Repository

```bash
git clone https://github.com/PrathamSharma-2811/ChatBot_Llama_Index_GenAi.git
cd ChatBot_Llama_Index_GenAi
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the root directory and add your HuggingFace API token.

```env
HF_API_TOKEN=your_huggingface_api_token
```

### Prepare Data

Place your document files in the `data` directory. The chatbot will use these documents to retrieve relevant information.

### Run the Application

```bash
python app.py
```

The chatbot will be accessible at `http://localhost:5003`.

## Llama Index and Llama3

### Llama Index

Llama Index is a tool for creating and managing an index of documents. It facilitates efficient document retrieval based on the content, allowing the chatbot to quickly find relevant information.

### Llama3

Llama3 is a pre-trained NLP model from HuggingFace used for generating intelligent responses. In this project, the `meta-llama/Meta-Llama-3-8B-Instruct` model is utilized for its large context window and high accuracy in understanding and responding to user queries.

## File Structure

- `app.py`: Main application file containing Flask routes and logic.
- `templates/`: Directory containing HTML templates for the web interface.
- `static/`: Directory for static files like CSS and JavaScript.
- `data/`: Directory for storing document files used by the chatbot.
- `requirements.txt`: List of Python dependencies.
- `suggestions.json`: JSON file containing predefined suggestions for commonly asked questions.

## Usage

1. **Ask Questions**: Users can interact with the chatbot by typing their questions in the input field.
2. **Get Suggestions**: The chatbot provides suggestions for commonly asked questions based on user input.
3. **Receive Email Notifications**: When a query is submitted, an email notification is sent to the specified recipient.

# Disclaimer

This project was built as a learning project during my internship period.I built this project from scratch.The chatbot and its functionalities are intended for educational purposes only. While efforts have been made to ensure the accuracy and reliability of the information provided by the chatbot, it should not be considered as an official source of information for Omaxe Pvt Ltd. For official information, please refer to the Omaxe website or contact the company directly. Use of this chatbot is at your own risk. The creators of this project are not responsible for any inaccuracies or issues that may arise from its use.

## Contact

For any issues or questions, please reach out to the me at [prathamnov@gmail.com].

---
