"""
DecisionIQ — RAG Evaluation Test Suite

This file performs small, repeatable tests for the
retrieval pipeline.

It checks:
1. Relevant retrieval
2. Cross-document retrieval
3. Unrelated query rejection
4. Similarity threshold behaviour
5. Source metadata preservation
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# TEST DATA
# =========================================================

# Simulated chunks representing multiple documents.
# In the real DecisionIQ application, these chunks come
# from uploaded PDFs.

chunks = [

    {
        "chunk_id": 1,
        "document": "Python_Notes.pdf",
        "page": 3,
        "text": (
            "A tuple is an ordered and immutable collection "
            "in Python. Tuples support indexing and slicing."
        )
    },

    {
        "chunk_id": 2,
        "document": "Python_Notes.pdf",
        "page": 4,
        "text": (
            "Tuples can contain duplicate values and can be "
            "used as dictionary keys when their elements are hashable."
        )
    },

    {
        "chunk_id": 3,
        "document": "Python_Sets.pdf",
        "page": 2,
        "text": (
            "A set is an unordered collection of unique elements."
        )
    },

    {
        "chunk_id": 4,
        "document": "Python_Sets.pdf",
        "page": 3,
        "text": (
            "Sets support operations such as union, intersection "
            "and difference."
        )
    }
]


# =========================================================
# MOCK EMBEDDINGS
# =========================================================

# These vectors simulate the output of an embedding model.
# They allow us to test the retrieval logic independently.

chunk_embeddings = np.array([

    [0.90, 0.10, 0.00],

    [0.85, 0.15, 0.00],

    [0.00, 0.90, 0.10],

    [0.00, 0.85, 0.15]

])


# =========================================================
# RETRIEVAL FUNCTION
# =========================================================

def retrieve(
    query_embedding,
    chunks,
    chunk_embeddings,
    top_k=3,
    threshold=0.35
):
    """
    Retrieve the most relevant chunks.

    Steps:
    1. Calculate cosine similarity.
    2. Sort chunks by score.
    3. Select top-k.
    4. Remove chunks below threshold.
    """

    similarity_scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    scored_chunks = list(
        zip(chunks, similarity_scores)
    )

    scored_chunks.sort(
        key=lambda item: item[1],
        reverse=True
    )

    top_chunks = scored_chunks[:top_k]

    relevant_chunks = [
        item
        for item in top_chunks
        if item[1] >= threshold
    ]

    return relevant_chunks


# =========================================================
# TEST 1 — RELEVANT RETRIEVAL
# =========================================================

def test_relevant_retrieval():

    query_embedding = np.array([
        [0.95, 0.05, 0.00]
    ])

    results = retrieve(
        query_embedding,
        chunks,
        chunk_embeddings
    )

    assert len(results) > 0

    top_chunk = results[0][0]

    assert top_chunk["document"] == "Python_Notes.pdf"

    print("[PASS] Relevant retrieval")


# =========================================================
# TEST 2 — CROSS-DOCUMENT RETRIEVAL
# =========================================================

def test_cross_document_retrieval():

    query_embedding = np.array([
        [0.05, 0.95, 0.00]
    ])

    results = retrieve(
        query_embedding,
        chunks,
        chunk_embeddings
    )

    assert len(results) > 0

    top_chunk = results[0][0]

    assert top_chunk["document"] == "Python_Sets.pdf"

    print("[PASS] Cross-document retrieval")


# =========================================================
# TEST 3 — UNRELATED QUERY REJECTION
# =========================================================

def test_unrelated_query():

    query_embedding = np.array([
        [0.00, 0.00, 1.00]
    ])

    results = retrieve(
        query_embedding,
        chunks,
        chunk_embeddings,
        threshold=0.90
    )

    assert len(results) == 0

    print("[PASS] Unrelated query rejection")


# =========================================================
# TEST 4 — THRESHOLD FILTERING
# =========================================================

def test_threshold_filtering():

    # This query is intentionally different from
    # the Python tuple/set embeddings.
    #
    # Therefore, its similarity with the stored chunks
    # should remain below the very high threshold.

    query_embedding = np.array([
        [0.50, 0.50, 0.70]
    ])

    results = retrieve(
        query_embedding,
        chunks,
        chunk_embeddings,
        threshold=0.99
    )

    assert len(results) == 0

    print("[PASS] Similarity threshold filtering")


# =========================================================
# TEST 5 — SOURCE METADATA
# =========================================================

def test_source_metadata():

    query_embedding = np.array([
        [0.90, 0.10, 0.00]
    ])

    results = retrieve(
        query_embedding,
        chunks,
        chunk_embeddings
    )

    assert len(results) > 0

    chunk = results[0][0]

    assert "document" in chunk
    assert "page" in chunk
    assert "text" in chunk

    print("[PASS] Source metadata preservation")


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("DecisionIQ RAG Evaluation")
    print("=" * 55)
    print()

    test_relevant_retrieval()

    test_cross_document_retrieval()

    test_unrelated_query()

    test_threshold_filtering()

    test_source_metadata()

    print()
    print("-" * 55)
    print("ALL TESTS PASSED")
    print("-" * 55)