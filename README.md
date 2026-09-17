# DecisionIQ — Evidence-Grounded AI Decision Assistant

DecisionIQ is an AI-powered document question-answering assistant built using
Retrieval-Augmented Generation (RAG).

It allows users to upload one or multiple PDF documents and ask questions
about their content. DecisionIQ retrieves relevant evidence from the uploaded
documents and uses that evidence to generate grounded answers with source
references.


## 🎯 Problem Statement

Traditional AI chatbots can sometimes generate answers that are not supported
by the user's documents.

DecisionIQ addresses this problem using an evidence-grounded RAG pipeline:

PDF → Text Extraction → Chunking → Embeddings → Semantic Retrieval
→ Context Building → LLM → Grounded Answer

The system is designed to reduce unsupported answers by requiring the LLM to
answer using retrieved document context.


## ✨ Features

- 📄 Upload single or multiple PDF documents
- 🔍 Semantic search using vector embeddings
- 🧩 Hybrid sentence-aware document chunking
- 📚 Cross-document retrieval
- 🎯 Top-K relevant chunk retrieval
- 🛑 Similarity threshold for irrelevant questions
- 🤖 Gemini-powered answer generation
- 📑 Source/page references for generated answers
- ⚡ Cached embedding model and document embeddings
- 🧪 RAG retrieval evaluation tests
- 📱 Mobile-friendly question submission
- 🎨 Custom dark UI with DecisionIQ branding
- 🔐 Secure API key management using Streamlit secrets


## 🧠 How DecisionIQ Works

### 1. PDF Upload

The user uploads one or more PDF documents.

DecisionIQ extracts text from every page using PyMuPDF.

### 2. Page-Aware Hybrid Chunking

The extracted content is divided into smaller chunks using a hybrid
chunking strategy that considers paragraph and sentence boundaries.

Each chunk preserves metadata such as:

- Chunk ID
- Page number
- Chunk text

### 3. Embeddings

Each document chunk is converted into a numerical vector using:

`all-MiniLM-L6-v2`

The model generates 384-dimensional embeddings.

### 4. Semantic Retrieval

When the user asks a question, the question is also converted into an
embedding.

Cosine similarity is used to compare the question embedding with document
chunk embeddings.

The most relevant chunks are selected using Top-K retrieval.

### 5. Similarity Threshold

DecisionIQ applies a similarity threshold to filter out unrelated questions.

If the retrieved evidence is not sufficiently relevant, the system does not
generate a document-grounded answer.

Instead, it informs the user that there is not enough relevant evidence in
the uploaded document.

### 6. Context Building

The relevant chunks are combined into structured context along with their
source page information.

### 7. Grounded Generation

The retrieved context is passed to Gemini with instructions to:

- Use only the provided context
- Avoid unsupported assumptions
- Avoid inventing facts
- Clearly state when evidence is insufficient
- Mention source pages when possible

### 8. Final Answer

The user receives an answer based on the retrieved document evidence along
with relevant source pages.


---

## 🏗️ Architecture

```text
                 ┌──────────────────┐
                 │    PDF Upload    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │     PyMuPDF      │
                 │  Text Extraction │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Hybrid Chunking  │
                 │ Paragraph +      │
                 │ Sentence Aware   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Sentence         │
                 │ Transformers     │
                 │ 384D Embeddings  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Semantic Search  │
                 │ Cosine Similarity│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Top-K Retrieval  │
                 │ + Threshold      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Context Builder  │
                 │ + Page Sources   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │      Gemini      │
                 │ Grounded LLM     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Answer + Sources │
                 └──────────────────┘

```
---

## 🧪 Evaluation

DecisionIQ includes a retrieval evaluation suite to validate the core RAG
pipeline.

The evaluation checks:

- Relevant document retrieval
- Cross-document retrieval
- Unrelated query rejection
- Similarity threshold filtering
- Source metadata preservation

Run the evaluation with:

```bash
python evaluation_test.py
```

## 🔐 Grounding & Hallucination Control

DecisionIQ uses multiple mechanisms to improve the reliability of
document-based answers:

1. Semantic retrieval
2. Top-K relevant chunk selection
3. Similarity threshold filtering
4. Source-aware context construction
5. Explicit grounding instructions for the LLM
6. Insufficient-evidence handling

If the uploaded document does not contain enough relevant evidence,
DecisionIQ does not generate a document-grounded answer.

Instead, the system informs the user that there is not enough relevant
evidence in the uploaded document.


---

## 🔮 Future Improvements

Potential improvements for future versions include:

- 🖼️ Image and camera-based document questioning
- 💾 Persistent vector database integration
- 🔄 Incremental document indexing
- 📈 Advanced RAG evaluation metrics
- 🔐 Authentication and user sessions
- 🧠 Reranking for improved retrieval quality
- 📊 Retrieval and answer quality dashboards
- ☁️ Scalable cloud deployment



---

## 👨‍💻 Author

**Dheeraj Kumar**

Computer Science Engineering Graduate  
AI/ML & Backend Developer

### Technologies

Python · RAG · LLMs · Machine Learning · Java · Spring Boot · PostgreSQL · Docker




---

## 📌 Project Status

**Version:** 1.0

DecisionIQ is a working RAG application with multi-PDF support,
semantic retrieval, retrieval evaluation, grounded Gemini generation,
and source attribution.
