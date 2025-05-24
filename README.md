# Mentat

An executive coach with a cool name.

> "Every hammer has the innate capacity to strike a nail. Every human mind has the innate capacity for greatness."

## Overview

This project provides a Python-based solution for working with ChromaDB, a vector database for AI applications. It helps you store, retrieve, and manage vector embeddings efficiently.

## Features

- Simple and intuitive API for ChromaDB operations
- Efficient vector storage and retrieval
- Support for semantic search capabilities
- Customizable embedding configurations

## Installation
bash
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt

## Usage

Basic example of using the ChromaDB interface:

python from chroma_db import ChromaDBClient
# Initialize the client
client = ChromaDBClient()
# Store embeddings
client.store_embeddings(documents, embeddings)
# Query similar documents
results = client.query_similar(query_embedding, n_results=5)


## Requirements

- Python 3.13+
- Dependencies listed in requirements.txt

## License

[License Type] - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.