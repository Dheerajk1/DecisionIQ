import streamlit as st
import pymupdf
import re
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DecisionIQ",
    page_icon="◈",
    layout="wide"
)


# =========================================================
# CUSTOM UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       DECISIONIQ — FINAL UI POLISH
       ===================================================== */

    /* ---------- Remove Streamlit's top chrome ---------- */

    [data-testid="stHeader"] {
        display: none;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }


    /* ---------- Main application ---------- */

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 8% 2%,
                rgba(124, 58, 237, 0.17),
                transparent 27%
            ),
            radial-gradient(
                circle at 94% 4%,
                rgba(6, 182, 212, 0.14),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(59, 130, 246, 0.055),
                transparent 35%
            ),
            #0b0d12;
    }


    .block-container {
        max-width: 1120px;
        padding-top: 1.35rem;
        padding-bottom: 3.5rem;
    }


    /* =====================================================
       DECISIONIQ BRAND
       ===================================================== */

    .brand-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }


    .brand-logo {
        width: 60px;
        height: 60px;
        border-radius: 18px;

        display: flex;
        align-items: center;
        justify-content: center;

        background:
            linear-gradient(
                135deg,
                #7c3aed 0%,
                #4f7cff 48%,
                #06b6d4 100%
            );

        color: white;
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: -1px;

        box-shadow:
            0 10px 32px rgba(124, 58, 237, 0.30),
            0 5px 20px rgba(6, 182, 212, 0.13);
    }


    .brand-name {
        font-size: 3.25rem;
        font-weight: 850;
        letter-spacing: -1.8px;
        line-height: 1;
    }


    .brand-name span {
        background:
            linear-gradient(
                90deg,
                #a78bfa,
                #22d3ee
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .subtitle {
        font-size: 1.15rem;
        color: #aeb8c8;

        margin-top: 0.72rem;
        margin-bottom: 1.05rem;
        margin-left: 76px;
    }


    /* ---------- Capability badges ---------- */

    .hero-badges {
        margin-left: 76px;
        margin-bottom: 2.55rem;
    }


    .hero-badge {
        display: inline-block;

        padding: 0.34rem 0.78rem;
        margin-right: 0.38rem;
        margin-bottom: 0.3rem;

        border: 1px solid rgba(96, 165, 250, 0.23);
        border-radius: 999px;

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.10),
                rgba(6, 182, 212, 0.07)
            );

        color: #b9c8dc;
        font-size: 0.77rem;

        box-shadow:
            0 5px 18px rgba(0, 0, 0, 0.08);
    }


    /* =====================================================
       SECTION TITLES
       ===================================================== */

    .section-title {
        font-size: 1.38rem;
        font-weight: 750;
        letter-spacing: -0.2px;

        margin-top: 1.55rem;
        margin-bottom: 0.75rem;
    }


    /* =====================================================
       UPLOAD AREA
       ===================================================== */

    [data-testid="stFileUploader"] {
        background:
            linear-gradient(
                145deg,
                rgba(34, 37, 49, 0.88),
                rgba(24, 27, 37, 0.86)
            );

        border: 1px solid rgba(148, 163, 184, 0.17);
        border-radius: 16px;

        padding: 0.38rem 0.55rem;

        box-shadow:
            0 16px 44px rgba(0, 0, 0, 0.15);
    }


    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: 1px dashed rgba(96, 165, 250, 0.31) !important;
        border-radius: 12px !important;
    }


    /* =====================================================
       QUESTION FORM
       ===================================================== */

    [data-testid="stForm"] {
        border: 0 !important;
        padding: 0 !important;
    }


    [data-testid="stTextInput"] input {
        min-height: 58px !important;

        border-radius: 14px !important;

        background:
            linear-gradient(
                145deg,
                rgba(32, 35, 47, 0.96),
                rgba(27, 30, 41, 0.96)
            ) !important;

        border: 1px solid rgba(148, 163, 184, 0.17) !important;

        color: #f3f4f6 !important;

        padding-left: 1rem !important;

        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease,
            background 0.2s ease;
    }


    [data-testid="stTextInput"] input::placeholder {
        color: #98a4b8 !important;
        opacity: 1 !important;
    }


    /* Replace the red focus outline with a blue/cyan glow. */

    [data-testid="stTextInput"] input:focus {
        border-color: rgba(96, 165, 250, 0.72) !important;

        box-shadow:
            0 0 0 1px rgba(96, 165, 250, 0.25),
            0 0 22px rgba(6, 182, 212, 0.08) !important;

        background:
            linear-gradient(
                145deg,
                rgba(34, 38, 52, 0.98),
                rgba(27, 31, 43, 0.98)
            ) !important;
    }


    /* ---------- Ask button ---------- */

    [data-testid="stFormSubmitButton"] button {
        min-height: 58px;

        border: 1px solid rgba(129, 140, 248, 0.30) !important;
        border-radius: 14px !important;

        background:
            linear-gradient(
                135deg,
                #6d3ee8 0%,
                #4f7cff 48%,
                #0891b2 100%
            ) !important;

        color: white !important;
        font-weight: 700 !important;

        box-shadow:
            0 10px 26px rgba(79, 124, 255, 0.18);

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            filter 0.18s ease;
    }


    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 13px 30px rgba(79, 124, 255, 0.26);

        filter: brightness(1.08);
    }


    [data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0);
    }


    /* =====================================================
       ANSWER BOX
       ===================================================== */

    .answer-box {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.88),
                rgba(18, 25, 39, 0.93)
            );

        border: 1px solid rgba(96, 165, 250, 0.22);
        border-radius: 16px;

        padding: 1.5rem 1.6rem;

        margin-top: 0.8rem;

        line-height: 1.78;

        box-shadow:
            0 18px 52px rgba(0, 0, 0, 0.18);
    }


    /* =====================================================
       SOURCE / DOCUMENT CARDS
       ===================================================== */

    .source-card {
        display: inline-block;

        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.72),
                rgba(20, 29, 43, 0.72)
            );

        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 11px;

        padding: 0.67rem 1rem;

        margin-right: 0.6rem;
        margin-bottom: 0.6rem;

        transition:
            border-color 0.2s ease,
            transform 0.2s ease;
    }


    .source-card:hover {
        border-color: rgba(96, 165, 250, 0.35);
        transform: translateY(-1px);
    }


    /* =====================================================
       INFO BOX
       ===================================================== */

    .info-box {
        background:
            linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.25),
                rgba(37, 99, 235, 0.10)
            );

        border: 1px solid rgba(96, 165, 250, 0.28);
        border-radius: 14px;

        padding: 1.05rem 1.25rem;

        margin: 1rem 0;

        box-shadow:
            0 12px 35px rgba(37, 99, 235, 0.06);
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    [data-testid="stMetric"] {
        min-height: 105px;

        padding: 0.95rem 1rem;

        border-radius: 14px;

        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.68),
                rgba(17, 24, 39, 0.56)
            );

        border: 1px solid rgba(148, 163, 184, 0.14);

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.11);
    }


    [data-testid="stMetricLabel"] {
        color: #a5afbf !important;
    }


    [data-testid="stMetricValue"] {
        font-weight: 750 !important;
    }


    /* =====================================================
       EXPANDER
       ===================================================== */

    [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 12px !important;
        background: rgba(15, 18, 26, 0.36) !important;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;

        color: #7f8a9b;

        font-size: 0.86rem;

        margin-top: 3.2rem;
        padding-top: 1.35rem;

        border-top:
            1px solid
            rgba(148, 163, 184, 0.13);

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(124, 58, 237, 0.035),
                rgba(6, 182, 212, 0.035),
                transparent
            );

        padding-bottom: 0.25rem;
    }


    .footer b {
        color: #c4ccda;
    }


    .footer a {
        color: #67e8f9 !important;
        text-decoration: none !important;
        font-weight: 600;
    }


    .footer a:hover {
        color: #a5f3fc !important;
        text-decoration: underline !important;
    }


    /* =====================================================
       MOBILE RESPONSIVENESS
       ===================================================== */

    @media (max-width: 700px) {

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2.5rem;
        }

        .brand-container {
            gap: 13px;
        }

        .brand-logo {
            width: 52px;
            height: 52px;
            border-radius: 15px;
        }

        .brand-name {
            font-size: 2.45rem;
        }

        .subtitle {
            margin-left: 65px;
            font-size: 1rem;
            margin-top: 0.55rem;
        }

        .hero-badges {
            margin-left: 65px;
            margin-bottom: 2rem;
        }

        [data-testid="stFormSubmitButton"] button {
            margin-top: 0.45rem;
        }

        .footer {
            font-size: 0.80rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="brand-container">
    <div class="brand-logo">DI</div>
    <div class="brand-name">Decision<span>IQ</span></div>
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="subtitle">
    Evidence-Grounded AI Decision Assistant
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="hero-badges">
    <span class="hero-badge">Multi-PDF RAG</span>
    <span class="hero-badge">Semantic Search</span>
    <span class="hero-badge">Source Grounded</span>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# =========================================================
# LOAD GEMINI CLIENT
# =========================================================

def get_gemini_api_key():
    """Get Gemini API key from Streamlit secrets or env var."""

    # Local: .streamlit/secrets.toml
    # Deployment: Streamlit Cloud Secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Fallback for environments using an environment variable.
    return os.getenv("GEMINI_API_KEY")


@st.cache_resource
def load_gemini_client():
    """Create Gemini client without crashing if the key is missing."""

    api_key = get_gemini_api_key()

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


gemini_client = load_gemini_client()


# =========================================================
# PRODUCTION VALIDATION HELPERS
# =========================================================

MAX_PDF_SIZE_MB = 200


def validate_uploaded_files(uploaded_files):
    """
    Validate uploaded PDFs before processing.

    Returns:
        (valid_files, rejected_files)
    """

    valid_files = []
    rejected_files = []
    seen_names = set()

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name

        # Duplicate filename protection
        if file_name in seen_names:
            rejected_files.append(
                (file_name, "Duplicate filename.")
            )
            continue

        seen_names.add(file_name)

        # File type protection
        if not file_name.lower().endswith(".pdf"):
            rejected_files.append(
                (file_name, "Only PDF files are supported.")
            )
            continue

        # File size protection
        file_size_mb = (
            len(uploaded_file.getvalue())
            / (1024 * 1024)
        )

        if file_size_mb > MAX_PDF_SIZE_MB:
            rejected_files.append(
                (
                    file_name,
                    f"File is larger than "
                    f"{MAX_PDF_SIZE_MB} MB."
                )
            )
            continue

        valid_files.append(uploaded_file)

    return valid_files, rejected_files


def clean_user_query(query):
    """Normalize whitespace in the user's question."""

    if query is None:
        return ""

    return " ".join(
        query.strip().split()
    )


def safe_gemini_text(response):
    """Safely extract text from a Gemini response."""

    if response is None:
        return ""

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        return ""

    return text.strip()


# =========================================================
# CACHE DOCUMENT EMBEDDINGS
# =========================================================

@st.cache_data(show_spinner=False)
def create_embeddings(chunk_texts):

    """
    Create embeddings for document chunks.

    Streamlit caches the result based on chunk_texts.

    If the same chunks are used again,
    cached embeddings can be reused.
    """

    return model.encode(
        list(chunk_texts)
    )


# =========================================================
# HYBRID CHUNKING
# =========================================================

def hybrid_chunking(
    text,
    chunk_size=500,
    chunk_overlap_sentences=1
):
    """
    Hybrid Chunking Strategy

    1. Clean extracted PDF text.
    2. Split text into paragraphs.
    3. Split paragraphs into sentences.
    4. Build chunks without unnecessarily
       breaking sentences.
    5. Reuse complete previous sentence(s)
       as overlap.
    6. Keep chunks within chunk_size.

    If one sentence itself is larger than
    chunk_size, a safe character fallback
    is used.
    """

    chunks = []


    # -----------------------------------------------------
    # STEP 1 — NORMALIZE WHITESPACE
    # -----------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    ).strip()


    # -----------------------------------------------------
    # STEP 2 — SPLIT INTO PARAGRAPHS
    # -----------------------------------------------------

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )


    # -----------------------------------------------------
    # PROCESS EACH PARAGRAPH
    # -----------------------------------------------------

    for paragraph in paragraphs:

        paragraph = paragraph.strip()


        if not paragraph:
            continue


        # -------------------------------------------------
        # STEP 3 — SPLIT PARAGRAPH INTO SENTENCES
        # -------------------------------------------------

        sentences = []

        current_sentence = ""


        for character in paragraph:

            current_sentence += character


            if character in ".!?":

                sentence = current_sentence.strip()


                if sentence:

                    sentences.append(
                        sentence
                    )


                current_sentence = ""


        # Add remaining text
        if current_sentence.strip():

            sentences.append(
                current_sentence.strip()
            )


        # If no sentence boundary was detected
        if not sentences:

            sentences = [
                paragraph
            ]


        # -------------------------------------------------
        # STEP 4 — BUILD CHUNKS
        # -------------------------------------------------

        current_chunk = []

        current_length = 0


        for sentence in sentences:

            sentence_length = len(sentence)


            # ---------------------------------------------
            # CASE A — SENTENCE LARGER THAN CHUNK SIZE
            # ---------------------------------------------

            if sentence_length > chunk_size:

                # Save current chunk
                if current_chunk:

                    chunks.append({

                        "text": " ".join(
                            current_chunk
                        ),

                        "sentences":
                            current_chunk.copy()

                    })


                    current_chunk = []

                    current_length = 0


                # Safe character fallback
                for i in range(
                    0,
                    sentence_length,
                    chunk_size
                ):

                    piece = sentence[
                        i:i + chunk_size
                    ]


                    chunks.append({

                        "text": piece,

                        "sentences": [piece]

                    })


                continue


            # ---------------------------------------------
            # CASE B — SENTENCE FITS
            # ---------------------------------------------

            if (
                current_length == 0
                or
                current_length
                + sentence_length
                + 1
                <= chunk_size
            ):

                current_chunk.append(
                    sentence
                )


                current_length += (
                    sentence_length + 1
                )


            # ---------------------------------------------
            # CASE C — CHUNK FULL
            # ---------------------------------------------

            else:

                # Save current chunk
                chunks.append({

                    "text": " ".join(
                        current_chunk
                    ),

                    "sentences":
                        current_chunk.copy()

                })


                # -----------------------------------------
                # SENTENCE-BASED OVERLAP
                # -----------------------------------------

                overlap_sentences = (
                    current_chunk[
                        -chunk_overlap_sentences:
                    ]
                )


                current_chunk = (
                    overlap_sentences.copy()
                )


                current_length = len(
                    " ".join(
                        current_chunk
                    )
                )


                # -----------------------------------------
                # Try adding new sentence
                # -----------------------------------------

                if (
                    current_length
                    + sentence_length
                    + 1
                    <= chunk_size
                ):

                    current_chunk.append(
                        sentence
                    )


                    current_length += (
                        sentence_length + 1
                    )


                else:

                    current_chunk = [
                        sentence
                    ]


                    current_length = (
                        sentence_length + 1
                    )


        # -------------------------------------------------
        # STEP 5 — SAVE FINAL CHUNK
        # -------------------------------------------------

        if current_chunk:

            chunks.append({

                "text": " ".join(
                    current_chunk
                ),

                "sentences":
                    current_chunk.copy()

            })


    return chunks


# =========================================================
# PDF UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📄 Upload a document'
    '</div>',
    unsafe_allow_html=True
)


uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# =========================================================
# QUESTION INPUT
# =========================================================

st.markdown(
    '<div class="section-title">'
    '💬 Ask a question'
    '</div>',
    unsafe_allow_html=True
)


# A form provides an explicit submit button for desktop/mobile.
# Pressing Enter in the text field can also submit the form.
with st.form(
    key="question_form",
    clear_on_submit=False,
    border=False
):

    question_col, button_col = st.columns(
        [5.2, 1.25],
        vertical_alignment="bottom"
    )

    with question_col:
        question_input = st.text_input(
            "Question",
            placeholder=(
                "Example: Can we modify a tuple "
                "after creating it?"
            ),
            label_visibility="collapsed"
        )

    with button_col:
        ask_clicked = st.form_submit_button(
            "Ask DecisionIQ",
            use_container_width=True
        )


# Only process a question after the user submits it.
query = clean_user_query(question_input) if ask_clicked else ""


# =========================================================
# NO PDF MODE
# =========================================================

if not uploaded_files:

    if query:

        st.markdown(
            '<div class="section-title">'
            '✨ General AI Answer'
            '</div>',
            unsafe_allow_html=True
        )


        if gemini_client is None:
            st.error(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to .streamlit/secrets.toml "
                "or your deployment's Secrets settings."
            )
            st.stop()

        with st.spinner(
            "Thinking..."
        ):

            try:

                response = (
                    gemini_client
                    .models
                    .generate_content(
                        model="gemini-3.6-flash",
                        contents=query
                    )
                )


                answer_text = safe_gemini_text(response)

                if not answer_text:
                    st.error(
                        "Gemini returned an empty response. "
                        "Please try the question again."
                    )
                    st.stop()

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            except Exception as e:

                st.error(
                    "Something went wrong while "
                    f"contacting Gemini: {e}"
                )


    else:

        st.markdown(
            """
            <div class="info-box">

            📄 Upload a PDF for
            evidence-grounded answers,

            or 💬 ask a general question.

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# PDF MODE — MULTI DOCUMENT
# =========================================================

else:

    # =====================================================
    # VALIDATE UPLOADED FILES
    # =====================================================

    valid_files, rejected_files = validate_uploaded_files(
        uploaded_files
    )

    for file_name, reason in rejected_files:
        st.warning(
            f"{file_name}: {reason}"
        )

    if not valid_files:
        st.error(
            "No valid PDF files are available for analysis."
        )
        st.stop()

    # Process only validated files.
    uploaded_files = valid_files


    # =====================================================
    # CONFIGURATION
    # =====================================================

    chunk_size = 500
    chunk_overlap_sentences = 1


    # =====================================================
    # STORAGE FOR ALL UPLOADED DOCUMENTS
    # =====================================================

    documents = []
    chunks = []
    pages_data = []
    all_text = []


    # =====================================================
    # PROCESS EVERY UPLOADED PDF
    # =====================================================

    for uploaded_file in uploaded_files:

        try:

            # Open the current PDF from memory.
            document = pymupdf.open(
                stream=uploaded_file.getvalue(),
                filetype="pdf"
            )

        except Exception as e:

            st.error(
                f"Unable to open {uploaded_file.name}. "
                f"Please check that it is a valid PDF."
            )
            continue


        # -------------------------------------------------
        # Store document-level metadata
        # -------------------------------------------------

        documents.append({
            "name": uploaded_file.name,
            "pages": document.page_count
        })


        # =================================================
        # PAGE-WISE TEXT EXTRACTION
        # =================================================

        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_text = page.get_text().strip()

            # Skip pages that contain no readable text.
            if not page_text:
                continue

            # Store complete text for statistics/debugging.
            all_text.append(page_text)

            # Store traceable page metadata.
            pages_data.append({
                "document": uploaded_file.name,
                "page": page_number,
                "text": page_text
            })


            # =================================================
            # HYBRID CHUNKING
            # =================================================

            page_chunks = hybrid_chunking(
                page_text,
                chunk_size=chunk_size,
                chunk_overlap_sentences=chunk_overlap_sentences
            )


            # =================================================
            # ADD CHUNKS + DOCUMENT/PAGE METADATA
            # =================================================

            for page_chunk in page_chunks:

                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "document": uploaded_file.name,
                    "page": page_number,
                    "text": page_chunk["text"]
                })


        # Close the current PDF after extraction.
        document.close()


    # =====================================================
    # EMPTY PDF PROTECTION
    # =====================================================

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]


    if not chunk_texts:

        st.error(
            "No readable text was found in the uploaded PDF(s)."
        )
        st.stop()


    # =====================================================
    # DOCUMENT STATISTICS
    # =====================================================

    total_documents = len(documents)

    total_pages = sum(
        document["pages"]
        for document in documents
    )

    full_text = "\n".join(all_text)


    # =====================================================
    # BASIC PARAGRAPHS
    # =====================================================

    paragraphs = []

    for text_block in all_text:

        for paragraph in text_block.split("\n"):

            paragraph = paragraph.strip()

            if paragraph:
                paragraphs.append(paragraph)


    # =====================================================
    # DOCUMENT OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Document Overview'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Documents",
            total_documents
        )


    with col2:

        st.metric(
            "Pages",
            total_pages
        )


    with col3:

        st.metric(
            "Chunks",
            len(chunks)
        )


    with col4:

        st.metric(
            "Embeddings",
            "384D"
        )


    # =====================================================
    # UPLOADED DOCUMENTS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📚 Uploaded Documents'
        '</div>',
        unsafe_allow_html=True
    )


    document_html = ""

    for doc in documents:

        document_html += (
            '<div class="source-card">'
            f'📄 {doc["name"]} '
            f'• {doc["pages"]} pages'
            '</div>'
        )


    st.markdown(
        document_html,
        unsafe_allow_html=True
    )


    # =====================================================
    # CREATE / LOAD CACHED EMBEDDINGS
    # =====================================================

    with st.spinner(
        "Preparing document embeddings..."
    ):

        chunk_embeddings = create_embeddings(
            tuple(chunk_texts)
        )


    # =====================================================
    # RETRIEVAL CONFIGURATION
    # =====================================================

    # Define these before the query check because Developer
    # Details also displays these values when no question exists.
    top_k = 3
    similarity_threshold = 0.35


    # =====================================================
    # RETRIEVAL ACROSS ALL DOCUMENTS
    # =====================================================

    if query:

        with st.spinner(
            "Searching across all documents..."
        ):

            # -------------------------------------------------
            # QUERY EMBEDDING
            # -------------------------------------------------

            query_embedding = model.encode(
                [query]
            )


            # -------------------------------------------------
            # COSINE SIMILARITY
            # -------------------------------------------------

            similarity_scores = cosine_similarity(
                query_embedding,
                chunk_embeddings
            )[0]


            # -------------------------------------------------
            # COMBINE CHUNKS + SCORES
            # -------------------------------------------------

            scored_chunks = list(
                enumerate(
                    zip(
                        chunks,
                        similarity_scores
                    )
                )
            )


            # -------------------------------------------------
            # SORT BY SIMILARITY
            # -------------------------------------------------

            scored_chunks.sort(
                key=lambda x: x[1][1],
                reverse=True
            )


            # -------------------------------------------------
            # TOP-K RETRIEVAL
            # -------------------------------------------------

            top_chunks = scored_chunks[:top_k]


            # -------------------------------------------------
            # SIMILARITY THRESHOLD
            # -------------------------------------------------

            relevant_chunks = [
                item
                for item in top_chunks
                if item[1][1] >= similarity_threshold
            ]


        # =====================================================
        # NO RELEVANT EVIDENCE
        # =====================================================

        if not relevant_chunks:

            st.markdown(
                '<div class="section-title">'
                '🔎 Document Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            st.warning(
                "No sufficiently relevant evidence was found "
                "in the uploaded PDF(s)."
            )

            if top_chunks:

                st.caption(
                    f"Best similarity score: "
                    f"{top_chunks[0][1][1]:.4f} "
                    f"| Threshold: "
                    f"{similarity_threshold:.2f}"
                )


        # =====================================================
        # RELEVANT EVIDENCE FOUND
        # =====================================================

        else:

            # -------------------------------------------------
            # CONTEXT BUILDER
            # -------------------------------------------------

            context_parts = []

            for item in relevant_chunks:

                chunk, score = item[1]

                formatted_chunk = (
                    f"[Source: {chunk['document']}, "
                    f"Page {chunk['page']}]\n"
                    f"{chunk['text']}"
                )

                context_parts.append(
                    formatted_chunk
                )


            context = "\n\n---\n\n".join(
                context_parts
            )


            # -------------------------------------------------
            # GROUNDED PROMPT
            # -------------------------------------------------

            grounded_prompt = f"""
You are DecisionIQ, an evidence-grounded AI assistant.

Answer the user's question using ONLY the provided context.

Important rules:

1. Use only information present in the context.
2. Do not use your general knowledge to fill missing information.
3. Do not invent facts.
4. If the context does not contain enough information,
   clearly say that there is not enough evidence in the
   uploaded documents.
5. Keep the answer concise and directly relevant.
6. When possible, mention the source document and page
   supporting the answer.

Context:
{context}

Question:
{query}

Answer:
"""


            # -------------------------------------------------
            # GEMINI GENERATION
            # -------------------------------------------------

            if gemini_client is None:

                st.error(
                    "Gemini API key is not configured. "
                    "Add GEMINI_API_KEY to .streamlit/secrets.toml "
                    "or your deployment's Secrets settings."
                )
                st.stop()


            with st.spinner(
                "Generating an evidence-grounded answer..."
            ):

                try:

                    response = (
                        gemini_client
                        .models
                        .generate_content(
                            model="gemini-3.6-flash",
                            contents=grounded_prompt
                        )
                    )

                except Exception as e:

                    st.error(
                        "Something went wrong while "
                        f"generating the answer: {e}"
                    )
                    st.stop()


            # =================================================
            # FINAL ANSWER
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '✨ DecisionIQ Answer'
                '</div>',
                unsafe_allow_html=True
            )

            answer_text = safe_gemini_text(response)

            if not answer_text:
                st.error(
                    "Gemini returned an empty response. "
                    "Please try the question again."
                )
                st.stop()

            st.markdown(
                f"""
                <div class="answer-box">
                {answer_text}
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # SOURCE PAGES
            # =================================================

            st.markdown(
                '<div class="section-title">'
                '📚 Sources'
                '</div>',
                unsafe_allow_html=True
            )


            # Unique document + page pairs.
            source_locations = sorted({
                (
                    item[1][0]["document"],
                    item[1][0]["page"]
                )
                for item in relevant_chunks
            })


            source_html = ""

            for document_name, page in source_locations:

                source_html += (
                    '<div class="source-card">'
                    f'📄 {document_name} • Page {page}'
                    '</div>'
                )


            st.markdown(
                source_html,
                unsafe_allow_html=True
            )


    # =====================================================
    # NO QUESTION
    # =====================================================

    else:

        st.markdown(
            """
            <div class="info-box">

            💬 Enter a question to search
            across the uploaded documents.

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # DEVELOPER DETAILS
    # =====================================================

    with st.expander(
        "🔧 Developer Details"
    ):

        st.write(
            "Gemini API:",
            "Configured" if gemini_client is not None
            else "Not configured"
        )

        st.write(
            "Documents:",
            total_documents
        )

        st.write(
            "Pages:",
            total_pages
        )

        st.write(
            "Paragraphs:",
            len(paragraphs)
        )

        st.write(
            "Chunks:",
            len(chunks)
        )

        st.write(
            "Total characters:",
            len(full_text)
        )

        st.write(
            "Embedding dimensions:",
            len(chunk_embeddings[0])
        )

        st.write(
            "Chunk size:",
            chunk_size
        )

        st.write(
            "Sentence overlap:",
            f"{chunk_overlap_sentences} sentence"
        )

        st.write(
            "Top-K retrieval:",
            top_k
        )

        st.write(
            "Similarity threshold:",
            similarity_threshold
        )

        st.write(
            "Maximum PDF size:",
            f"{MAX_PDF_SIZE_MB} MB"
        )


        # =================================================
        # PAGE METADATA
        # =================================================

        st.markdown(
            "### 📄 Page Metadata"
        )

        for page_data in pages_data:

            with st.expander(
                f"{page_data['document']} • "
                f"Page {page_data['page']}"
            ):

                st.text(
                    page_data["text"]
                )


        # =================================================
        # COMPLETE EXTRACTED TEXT
        # =================================================

        st.markdown(
            "### 📝 Complete Extracted Text"
        )

        st.text_area(
            "Extracted Text",
            full_text,
            height=300
        )


        # =================================================
        # EXTRACTED PARAGRAPHS
        # =================================================

        st.markdown(
            "### 📑 Extracted Paragraphs"
        )

        for i, paragraph in enumerate(
            paragraphs
        ):

            st.write(
                f"Paragraph {i + 1}"
            )

            st.text_area(
                f"Paragraph {i + 1}",
                paragraph,
                height=80,
                key=f"paragraph_{i}"
            )


        # =================================================
        # GENERATED CHUNKS
        # =================================================

        st.markdown(
            "### 🧩 Generated Chunks"
        )

        for chunk in chunks:

            st.write(
                f"Chunk {chunk['chunk_id']} "
                f"| {chunk['document']} "
                f"| Page {chunk['page']}"
            )

            st.text_area(
                f"Chunk {chunk['chunk_id']}",
                chunk["text"],
                height=120,
                key=f"chunk_{chunk['chunk_id']}"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">DecisionIQ v1.0 · by Dheeraj Kumar · <a href="https://www.linkedin.com/in/dheeraj-kumar-dk1001/" target="_blank">LinkedIn ↗</a></div>
    """,
    unsafe_allow_html=True
)