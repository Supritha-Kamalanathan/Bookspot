# Local RAG Model for Book Recommendation 

## Overview

This project implements a local RAG model from scratch for a book recommendation system. It aims to explore and understand the mechanics of RAG models by integrating text chunking, embedding generation, and language models. The system utilizes a PostgreSQL database for data storage and retrieval, and Streamlit for the web interface.

## Features

- **Custom Chunking**: Splits book descriptions into manageable chunks with options for chunk size, overlap, and secondary chunking based on regex patterns.
- **Text Embeddings**: Generates embeddings for book chunks and user queries using a pre-trained transformer model.
- **Similarity Calculation**: Computes cosine similarity between query embeddings and stored chunk embeddings to identify the most relevant book chunks.
- **Dynamic Recommendations**: Retrieves top matching book chunks and generates recommendations using a language model.
- **Web Interface**: Provides an interactive web interface using Streamlit for user queries and recommendations.
![Screenshot (521)](https://github.com/user-attachments/assets/bed30195-b8c1-47d3-a6cd-ecb43b4383a2)
