# Agentic AI RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) application designed to answer questions from the Agentic AI eBook. It uses a graph-based workflow to ensure strictly grounded responses.

## Tech Stack

- **Framework:** LangGraph (StateGraph)
- **Database:** ChromaDB (Vector Store)
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **LLM:** DeepSeek (via OpenRouter)
- **UI:** Streamlit

## Architecture

The system follows a two-node pipeline:

- **Retrieve Node:** Expands the user query and fetches relevant chunks from the vector database.
- **Generate Node:** Uses a strict prompt to generate answers based only on the provided context, providing a confidence score.

## Setup Instructions

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Configure Environment

1. Create a file named `.env` in the root directory.
2. Add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### Ingest Data

1. Place your PDF in the `./data/` folder (default: `Ebook-Agentic-AI.pdf`).
2. Run the ingestion script to create the vector database:

```bash
python ingest.py
```

### Run the App

Launch the Streamlit interface:

```bash
streamlit run app.py
```

## Sample Queries

- What is Konverge AI?
- Explain  the Shift from Reactive to Proactive Technology.
- what is A Journey into the Heart of Autonomous Intelligence 
- Defining Characteristics of an Agent.
- What are "Types of Atomic Agents"