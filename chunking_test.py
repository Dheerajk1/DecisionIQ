# ============================================
# DecisionIQ - Day 4
# Chunking Quality Experiment
# ============================================

# In this file we are testing different
# chunking strategies before changing app.py.
#
# We will compare:
#
# 1. Fixed Character Chunking
# 2. Paragraph-Aware Chunking
# 3. Hybrid Chunking
#
# Goal:
# Create chunks that:
# - preserve meaning
# - avoid cutting words/sentences unnecessarily
# - don't become extremely large
# ============================================


# =================================================
# SAMPLE DOCUMENT
# =================================================

text = """
Python is a high-level programming language.
It is widely used for web development, automation,
data science, artificial intelligence and machine learning.

Machine learning is a branch of artificial intelligence.
It allows computers to learn patterns from data
and make predictions or decisions.

This is a deliberately long paragraph for testing our
hybrid chunking strategy. In a real PDF document, a paragraph
can sometimes contain a large amount of information about a
single topic. If the paragraph becomes too large, keeping the
entire paragraph inside one chunk can reduce retrieval quality.
Our chunking system should therefore understand that the
paragraph is too large and split it into smaller pieces.
However, it should not randomly cut the paragraph in the
middle of a word or sentence. Instead, it should try to keep
complete sentences together while creating chunks that stay
within the maximum chunk size. This allows the embedding model
to receive meaningful pieces of information instead of random
pieces of text. Better chunks can lead to better retrieval
because each embedding represents a more focused piece of
information. This is particularly useful in Retrieval-Augmented
Generation systems where the retrieved chunks are later provided
to the language model as context for generating a grounded answer.

Deep learning is a subset of machine learning.
It uses neural networks with multiple layers
to learn complex patterns from large amounts of data.

Natural Language Processing, or NLP, focuses on
enabling computers to understand and process human language.
It is commonly used in chatbots, translation and text classification.
"""


# =================================================
# METHOD 1 — FIXED CHARACTER CHUNKING
# =================================================

def fixed_chunking(text, chunk_size=100, chunk_overlap=20):

    chunks = []

    # Example:
    #
    # chunk_size = 100
    # chunk_overlap = 20
    #
    # So every next chunk starts after:
    #
    # 100 - 20 = 80 characters

    step = chunk_size - chunk_overlap

    for i in range(0, len(text), step):

        chunk = text[i:i + chunk_size]

        chunks.append(chunk)

    return chunks


# =================================================
# METHOD 2 — PARAGRAPH-AWARE CHUNKING
# =================================================

def paragraph_chunking(text, chunk_size=500):

    chunks = []

    # First divide the document into paragraphs.
    #
    # "\n\n" means two newline characters.
    #
    # Our sample document has blank lines between
    # different topics.

    paragraphs = text.split("\n\n")

    current_chunk = ""

    for paragraph in paragraphs:

        # Remove unnecessary spaces/newlines
        paragraph = paragraph.strip()

        # Ignore empty paragraphs
        if not paragraph:
            continue

        # -----------------------------------------
        # If paragraph fits inside the chunk,
        # keep the paragraph together.
        # -----------------------------------------

        if len(current_chunk) + len(paragraph) <= chunk_size:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:

            # Current chunk is full.
            # Save it first.

            if current_chunk:
                chunks.append(current_chunk)

            # Start a new chunk with this paragraph
            current_chunk = paragraph

    # Save the final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# =================================================
# METHOD 3 — FINAL HYBRID CHUNKING
# =================================================

def hybrid_chunking(
    text,
    chunk_size=500,
    chunk_overlap_sentences=1
):
    """
    Final Hybrid Chunking Strategy

    Strategy:

    1. Split document into paragraphs.
    2. Small paragraphs are kept together.
    3. Large paragraphs are split into sentences.
    4. Chunks are built without breaking sentences.
    5. Previous complete sentence(s) are reused as overlap.
    6. Every chunk stays within chunk_size.
    """

    chunks = []

    # ------------------------------------------------
    # STEP 1 — Split document into paragraphs
    # ------------------------------------------------

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # ------------------------------------------------
        # STEP 2 — Convert paragraph into sentences
        # ------------------------------------------------
        #
        # This simple version handles:
        # .
        # !
        # ?
        #
        # We keep the punctuation.
        # ------------------------------------------------

        sentences = []

        current_sentence = ""

        for character in paragraph:

            current_sentence += character

            if character in ".!?":

                sentence = current_sentence.strip()

                if sentence:
                    sentences.append(sentence)

                current_sentence = ""

        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # ------------------------------------------------
        # STEP 3 — Build chunks from sentences
        # ------------------------------------------------

        current_chunk = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence)

            # --------------------------------------------
            # CASE 1
            # Sentence itself is larger than chunk_size
            # --------------------------------------------

            if sentence_length > chunk_size:

                # Save existing chunk first
                if current_chunk:

                    chunks.append(
                        " ".join(current_chunk)
                    )

                    current_chunk = []
                    current_length = 0

                # Extremely long sentence:
                # fallback to character splitting.
                #
                # This is rare, but protects us from
                # creating a chunk larger than 500 chars.

                for i in range(0, sentence_length, chunk_size):

                    chunks.append(
                        sentence[i:i + chunk_size]
                    )

                continue

            # --------------------------------------------
            # CASE 2
            # Sentence fits in current chunk
            # --------------------------------------------

            if (
                current_length == 0
                or current_length + sentence_length + 1 <= chunk_size
            ):

                current_chunk.append(sentence)

                current_length += (
                    sentence_length + 1
                )

            # --------------------------------------------
            # CASE 3
            # Current chunk is full
            # --------------------------------------------

            else:

                # Save current chunk
                chunks.append(
                    " ".join(current_chunk)
                )

                # ----------------------------------------
                # Create sentence-based overlap
                # ----------------------------------------

                overlap_sentences = current_chunk[
                    -chunk_overlap_sentences:
                ]

                # Start next chunk with complete
                # sentence(s), NOT characters.
                current_chunk = overlap_sentences.copy()

                current_length = len(
                    " ".join(current_chunk)
                )

                # ----------------------------------------
                # Add the new sentence
                # ----------------------------------------

                if (
                    current_length + sentence_length + 1
                    <= chunk_size
                ):

                    current_chunk.append(sentence)

                    current_length += (
                        sentence_length + 1
                    )

                else:

                    # If overlap + new sentence doesn't fit,
                    # remove overlap and start fresh.

                    current_chunk = [sentence]

                    current_length = (
                        sentence_length + 1
                    )

        # ------------------------------------------------
        # STEP 4 — Save final chunk of this paragraph
        # ------------------------------------------------

        if current_chunk:

            chunks.append(
                " ".join(current_chunk)
            )

    return chunks

# =================================================
# RUN ALL THREE METHODS
# =================================================

fixed_chunks = fixed_chunking(
    text,
    chunk_size=100,
    chunk_overlap=20
)

paragraph_chunks = paragraph_chunking(
    text,
    chunk_size=500
)

hybrid_chunks = hybrid_chunking(
    text,
    chunk_size=500,
    chunk_overlap_sentences=1
)


# =================================================
# PRINT FIXED CHUNKING
# =================================================

print("\n")
print("=" * 70)
print("METHOD 1 — FIXED CHARACTER CHUNKING")
print("=" * 70)

for i, chunk in enumerate(fixed_chunks, start=1):

    print(f"\n--- Chunk {i} ---")
    print(chunk)


# =================================================
# PRINT PARAGRAPH-AWARE CHUNKING
# =================================================

print("\n")
print("=" * 70)
print("METHOD 2 — PARAGRAPH-AWARE CHUNKING")
print("=" * 70)

for i, chunk in enumerate(paragraph_chunks, start=1):

    print(f"\n--- Chunk {i} ---")
    print(chunk)


# =================================================
# PRINT HYBRID CHUNKING
# =================================================

print("\n")
print("=" * 70)
print("METHOD 3 — HYBRID CHUNKING")
print("=" * 70)

for i, chunk in enumerate(hybrid_chunks, start=1):

    print(f"\n--- Chunk {i} ---")
    print(chunk)


# =================================================
# COMPARISON
# =================================================

print("\n")
print("=" * 70)
print("CHUNKING COMPARISON")
print("=" * 70)

print("Fixed chunks     :", len(fixed_chunks))
print("Paragraph chunks :", len(paragraph_chunks))
print("Hybrid chunks    :", len(hybrid_chunks))


# =================================================
# CHUNK SIZE ANALYSIS
# =================================================

print("\n")
print("=" * 70)
print("HYBRID CHUNK SIZE ANALYSIS")
print("=" * 70)

for i, chunk in enumerate(hybrid_chunks, start=1):

    print(
        f"Chunk {i}: "
        f"{len(chunk)} characters"
    )