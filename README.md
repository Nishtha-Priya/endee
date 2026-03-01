# HireReady – RAG-Based AI Interview Simulator using Endee

## Overview

HireReady is a Retrieval-Augmented Generation (RAG) based AI Interview Simulator built on top of the Endee Vector Database.

The system generates role-specific technical interview questions and evaluates candidate answers by leveraging semantic retrieval over resume and job description data.

This project demonstrates a practical AI/ML use case where vector search is the core component of the pipeline.

---

## Problem Statement

Most interview preparation platforms generate generic questions that are not tailored to:

- The candidate’s resume  
- The specific job description  
- The technologies required for the role  

As a result, candidates do not receive personalized or context-aware interview practice.

HireReady solves this by:

- Converting resume and job description into vector embeddings  
- Storing them inside Endee Vector Database  
- Using semantic retrieval to generate context-aware interview questions  
- Evaluating answers based on retrieved technical context  

This makes the system dynamic, personalized, and role-aware.

---

## Practical AI Use Case Demonstrated

This project implements:

- Retrieval-Augmented Generation (RAG)
- Semantic Search using vector similarity
- Context-aware question generation
- Role-aware answer evaluation

Vector search is the core component of the system.  
Without Endee’s semantic retrieval, the system would not be able to generate context-specific interview questions.

---

## System Architecture

High-Level Flow:

Resume / Job Description  
→ Embedding Model (Sentence Transformers – MiniLM)  
→ Vector Storage in Endee (cosine similarity)  
→ Semantic Retrieval (top-k matches)  
→ FLAN-T5 LLM for question generation  
→ Answer evaluation logic  

---

## Technical Approach

### 1. Embedding Generation

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output dimension: 384
- Converts resume and JD into dense semantic vectors

### 2. Vector Database (Endee)

- Index: `interview_index`
- Dimension: 384
- Metric: Cosine Similarity
- Automatic index creation if not present
- Resume and JD embeddings stored using `upsert`
- Retrieval performed using `top_k` semantic search

### 3. Retrieval-Augmented Generation (RAG)

When a user requests interview questions:

1. Query is embedded  
2. Endee retrieves semantically relevant context  
3. Retrieved context is injected into LLM prompt  
4. FLAN-T5 generates advanced technical questions  

This ensures questions are grounded in candidate + role context.

### 4. Answer Evaluation

The system:

- Extracts technical keywords from retrieved context  
- Compares them against user response  
- Scores answer out of 10  
- Provides strengths and weaknesses  

---

---

## Future Improvements

While the current system demonstrates a complete RAG-based interview simulator using Endee as the vector database, several enhancements can be made:

### 1. Better Document Handling
Currently, the resume and job description are stored as single embeddings.  
Future versions can:
- Split documents into smaller meaningful sections  
- Store multiple embeddings per document  
- Improve retrieval precision  

### 2. Hybrid Search (Keyword + Vector)
Combine:
- Semantic similarity search  
- Simple keyword matching  
To improve retrieval relevance and control.

### 3. Improved Question Accuracy
Enhance question quality by:
- Retrieving more relevant context  
- Refining prompts given to the LLM  
- Filtering out weak or unrelated results  

### 4. Adaptive Interview Flow
Extend the system to:
- Ask follow-up questions dynamically  
- Adjust difficulty based on previous answers  

### 5. Scalable Deployment
Deploy the full system on cloud infrastructure for:
- Multi-user access  
- Better performance  
- Real-world usage  

---

## How Endee Is Used

Endee is the core infrastructure component of this project.

It is used to:

- Create a vector index (dimension=384, metric=cosine)
- Store resume and job description embeddings
- Perform semantic similarity search
- Retrieve relevant context for the RAG pipeline

The system automatically creates the index if it does not exist, ensuring evaluator-safe execution on a fresh database instance.

Endee runs locally via Docker.

---

## Tech Stack

- Python  
- Endee Vector Database  
- Sentence Transformers  
- HuggingFace Transformers (FLAN-T5)  
- Streamlit  
- Docker  

---

## Setup Instructions

### 1. Clone the Forked Repository

```bash
git clone https://github.com/Nishtha-Priya/endee.git
cd endee
```

### 2. Start Endee Server

```bash
docker compose up -d
```

Endee will run on:

```
http://localhost:8080
```

### 3. Setup Python Environment

```bash
cd HireReady
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will be available at:

```
http://localhost:8501
```

---

## Why This Project Demonstrates ML Engineering Skills

- Implements Retrieval-Augmented Generation from scratch  
- Uses semantic vector search as the core retrieval mechanism  
- Integrates an external vector database (Endee) properly  
- Handles automatic index creation for robustness  
- Combines embeddings, similarity search, and LLM generation  
- Modular architecture (embedding, vector store, RAG pipeline separation)  

This project demonstrates practical application of:

- Embeddings  
- Cosine similarity  
- Vector databases  
- Context-aware generation  
- ML system integration  

---

## Built On

This project is built on top of the official Endee Vector Database repository:

https://github.com/endee-io/endee
