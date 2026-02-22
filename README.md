# MoviePilot

MoviePilot is a Retrieval-Augmented Generation (RAG) search engine designed for movie datasets. It combines keyword-based retrieval with semantic search and also adds multimodal search, query enhancement and rag techniques to provide relevant movie discovery.

## Description

MoviePilot combines different traditional and modern information retrieval techniques from basic term matching to AI-driven retrieval:

* Keyword Search: Uses Okapi BM25 and TF-IDF for term matching.
* Semantic Search: Utilizes vector embeddings and chunking strategies (fixed and semantic) to understand the context of queries.
* Hybrid Search: Merges keyword and semantic results using Reciprocal Rank Fusion (RRF) and weighted scoring for balanced retrieval.
* Multimodal Search: Enables movie discovery using image-to-text and image-embedding capabilities.
* RAG: Integrates with Google's Gemini LLM to generate summaries, answer questions, and provide cited responses based on retrieved movie documents.

## Motivation

The primary motivation for this project was to gain hands-on experience with the core components of modern search systems. It serves as a practical learning opportunity for:

* Information Retrieval (IR): Understanding the mathematical foundations of keyword search, such as term frequency (TF) and inverse document frequency (IDF).
* RAG System Design: Learning how to build an end-to-end pipeline that connects a retrieval engine to a Large Language Model (LLM) to reduce hallucinations and provide grounded answers.
* Hybrid Strategies: Experimenting with combining semantic and keyword retrieval to overcome the limitations of each individual method.

## Quick Start

### 1. Prerequisites

Ensure you have Python 3.12+ and uv installed.

### 2. Setup Environment

Clone the repository:

```bash
git clone https://github.com/jg-qbig/MoviePilot
cd MoviePilot

```

Create a `.env` file in the root directory. Add your Google GenAI API key:

```env
GEMINI_API_KEY=your_api_key_here

```

### 3. Install Dependencies

You can install the project and its dependencies using `pip`:

```bash
uv sync --locked

```

---
## Contributing

Any pull requests or contributions are welcome.
