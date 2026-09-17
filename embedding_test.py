from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Embedding model load karna
# ---------------------------------------------------------

# Pre-trained model load kar rahe hain.
# Ye har text ko 384-dimensional vector mein convert karega.
model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# Sample document chunks
# ---------------------------------------------------------

# Maan lo ye chunks hamare PDF se aaye hain.
chunks = [
    "Python is a programming language.",
    "Python is used for software development.",
    "The weather is very pleasant today."
]


# ---------------------------------------------------------
# User Query
# ---------------------------------------------------------

# User ka question.
query = "Tell me about Python programming"


# ---------------------------------------------------------
# Chunks ko embeddings mein convert karna
# ---------------------------------------------------------

# Har chunk ko numerical vector mein convert kar rahe hain.
chunk_embeddings = model.encode(chunks)


# ---------------------------------------------------------
# Query ko embedding mein convert karna
# ---------------------------------------------------------

# User query ko bhi vector mein convert kar rahe hain.
query_embedding = model.encode([query])


# ---------------------------------------------------------
# Cosine Similarity calculate karna
# ---------------------------------------------------------

# Query ko har chunk ke saath compare kar rahe hain.
similarity_scores = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]


# ---------------------------------------------------------
# Similarity ke basis par chunks ko rank karna
# ---------------------------------------------------------

# enumerate() se hume chunk ka index aur score dono milenge.
scored_chunks = list(
    enumerate(zip(chunks, similarity_scores))
)


# Highest similarity score ko pehle laane ke liye
# descending order mein sort kar rahe hain.
scored_chunks.sort(
    key=lambda x: x[1][1],
    reverse=True
)


# ---------------------------------------------------------
# Top-K Retrieval
# ---------------------------------------------------------

# Hum sirf top 2 most relevant chunks retrieve karna chahte hain.
top_k = 2


print("\nQuery:", query)

print("\nTop Relevant Chunks:")
print("=" * 60)


# Top 2 chunks retrieve kar rahe hain.
for rank, (index, (chunk, score)) in enumerate(
    scored_chunks[:top_k],
    start=1
):

    print(f"Rank: {rank}")
    print(f"Original Chunk Number: {index + 1}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Chunk: {chunk}")

    print("-" * 60)