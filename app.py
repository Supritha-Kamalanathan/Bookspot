import numpy as np
from llama_cpp import Llama
import torch
from transformers import AutoModel, AutoTokenizer
import os
import psycopg2
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

# Connecting to database
conn = psycopg2.connect(
    dbname = db_name,
    user = db_user,
    password = db_password,
    host = db_host,
    port = db_port
)

cursor = conn.cursor()

model_name = "BAAI/bge-small-en-v1.5"

# Function to generate embeddings for a given text using the loaded model and tokenizer
def compute_embeddings(text):
    tokenizer_save_path = "model/tokenizer"
    model_save_path = "model/embedding"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    model = AutoModel.from_pretrained(model_save_path)

    inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True)

    # Temporarily disables gradient calculation which reduces memory usage and speeds up computation
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim = 1).squeeze()
    
    return embeddings.tolist()

# Function that computes embeddings of the query string, computes cosine similarity, and return top_k matches based on the scores
def compute_matches(query_str, top_k):
    query_str_embedding = np.array(compute_embeddings(query_str))
    scores = []

    cursor.execute("SELECT b_id, chunk_id, embeddings FROM chunks WHERE embeddings IS NOT NULL")
    rows = cursor.fetchall()

    book_ids = []
    chunk_ids = []
    chunk_embeddings = []

    for book_id, chunk_id, chunk_embedding in rows:
        book_ids.append(book_id)
        chunk_ids.append(chunk_id)
        chunk_embeddings.append(chunk_embedding)

    # Convert list of chunk embeddings to numpy array for batch processing
    chunk_embeddings_array = np.array(chunk_embeddings)

    # Normalize the query and chunk embeddings for cosine similarity calculation
    norm_query = np.linalg.norm(query_str_embedding)
    norm_chunk = np.linalg.norm(chunk_embeddings_array, axis = 1)

    # Prevent division by zero
    valid_norms = np.where(norm_query != 0 and norm_chunk != 0)

    # Compute cosine similarities in a batch
    cosine_similarities = np.zeros(len(chunk_embeddings_array))
    cosine_similarities[valid_norms] = np.dot(chunk_embeddings_array[valid_norms], query_str_embedding) / (norm_chunk[valid_norms] * norm_query)

    scores = [(book_ids[i], chunk_ids[i], cosine_similarities[i]) for i in range(len(book_ids))]
    top_results = sorted(scores, key = lambda item: item[2], reverse = True)[:top_k]

    return top_results

# Function to retrieve data of the top matches
def retrieve_data(matches):
    for match in matches:
        book_id = match[0]
        chunk_id = match[1]

        cursor.execute("SELECT text FROM chunks WHERE b_id = %s AND chunk_id = %s", (book_id, chunk_id))
        data = cursor.fetchall()

    return data

# Function to construct the prompt
def construct_prompt(system_prompt, retrieved_data, user_query):
    prompt = f"""{system_prompt}

    Here is the user's query:
    {user_query}

    Here is the retrieved context:
    {retrieved_data}

    """
    return prompt

llm = Llama(model_path="model/mistral-7b-instruct-v0.2.Q3_K_L.gguf")

# Streamlit frontend
st.title("Book Recommendation System")

query_str = st.text_input("Enter your query: ")

if st.button("Get recommendations"):
    if query_str:
        with st.spinner('Generating recommendations...'):
            matches = compute_matches(query_str=query_str, top_k=3)
            retrieved_data = retrieve_data(matches)

            system_prompt = """
                You are a knowledgeable and helpful book recommendation system. Your task is to provide book recommendations and insights based on the context provided. You should rely solely on the information given in the context to generate your responses. Do not include any information that is not present in the context. Focus on providing relevant and accurate recommendations or answers according to the book descriptions and details provided.
            """

            base_prompt = construct_prompt(system_prompt=system_prompt, retrieved_data=retrieved_data, user_query=query_str)
            formatted_prompt = f"Q: {base_prompt} A: "

            response = llm(formatted_prompt, max_tokens=500, temperature=0.1, top_k=40, echo=False, stream=False)

        st.write("Here is the recommendation: ")
        st.write(response["choices"][0]["text"])
    else:
        st.warning("Please enter a query!")

st.info("Type in your preferences, interests, or a specific book you're interested in, and get recommendations!")
