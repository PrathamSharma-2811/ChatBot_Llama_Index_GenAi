from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import Settings
from flask import Flask,jsonify
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import huggingface_hub
import posixpath
import torch
from flask import render_template, request,jsonify
import requests
from llama_index.core import PromptTemplate
import base64
import json
import os
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.llms import ChatMessage
import markdown
import re

# HF_HUB_CACHE
huggingface_hub.login(token="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")


documents=SimpleDirectoryReader("data").load_data()
app = Flask(__name__)

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# System prompt for LLMS
system_prompt = """<|SYSTEM|>
        You are customer support agent , based on the context provide answer the questions of user.
Context:\n {context}?\n
Question: \n{question}\n
"""


query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

# LLMS settings
llm2 = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Set settings
Settings.llm = llm2
Settings.embed_model = embed_model
Settings.chunk_size = 1024

# Index documents
index = VectorStoreIndex.from_documents(documents)
retrieval = index.as_retriever()

# Chat memory buffer
memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt,
    llm=llm2,
    verbose=False
)
def gen_response(question):
    
    response = chat_engine.chat(question)
   
    return str(response)

def clean_response(response):
    pattern = r'^\s*assistant\s*'

    # Use re.sub to replace the pattern with an empty string
    cleaned_response = re.sub(pattern, '', response)
    return cleaned_response

# Load suggestions from JSON
def load_suggestions_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        suggestions = json.load(file)
    return suggestions

predefined_suggestions = load_suggestions_from_json('suggestions.json')

# Route for suggestion
@app.route('/suggest', methods=['POST'])
def suggest():
    input_text = request.form['input_text'].lower()
    suggestions = get_suggestions(input_text)
    return jsonify(suggestions)

# Get suggestions
def get_suggestions(input_text):
    return [suggestion for suggestion in predefined_suggestions if input_text.lower() in suggestion.lower()]

# Route to save and send query
@app.route('/save_and_send', methods=['GET', 'POST'])
def save_and_send():
    email = request.form['email']
    category = request.form['category']
    query = request.form['taskdescription']
    send_email(email, category, query)
    return render_template('chat.html')

# Function to send email
def send_email(email, category, query):
    subject = f'New query from {category} category'
    sender_name = "User"
    sender_email = email
    recipient_email = "bigsecxxv@gmail.com"
    body = f'New query from {category} category: {query}'

    payload = {
        "sender": {
            "name": sender_name,
            "email": sender_email
        },
        "to": [
            {
                "email": recipient_email
            }
        ],
        "subject": subject,
        "htmlContent": body
    }

    

    response = requests.post(api_url, headers={"api-key": api_key}, json=payload)
    if response:
        scroll_to_contact = True if request.path.endswith('/sent') else False
        return render_template("chat.html", status="Successfully", scroll_to_contact=scroll_to_contact)
    else:
        scroll_to_contact = True if request.path.endswith('/sent') else False
        return render_template("chat.html", status="Successfully", scroll_to_contact=scroll_to_contact)

# Route for chat
@app.route("/get", methods=["GET", "POST"])
def chat():
    
    question = request.form.get("msg")
    response = gen_response(question)
    # print(response)
    cleaned_response = clean_response(response)
    html_response = markdown.markdown(cleaned_response)
    return jsonify({'response': html_response})
# Route for index
@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=False, port=5003)
