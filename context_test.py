# ---------------------------------------------------------
# DecisionIQ - Context Builder + Grounded Gemini Test
# ---------------------------------------------------------
#
# Is file ka purpose:
#
# Retrieved chunks ko structured context mein convert karna
# aur phir us context ko Gemini LLM ko dena.
#
# RAG pipeline:
#
# User Question
#       ↓
# Retrieval
#       ↓
# Top Relevant Chunks
#       ↓
# Context Builder
#       ↓
# Grounded Prompt
#       ↓
# Gemini LLM
#       ↓
# Final Answer
# ---------------------------------------------------------


# ---------------------------------------------------------
# IMPORT GEMINI SDK
# ---------------------------------------------------------
#
# google-genai SDK Gemini API ke saath communicate
# karne ke liye use ho raha hai.
# ---------------------------------------------------------

from google import genai


# ---------------------------------------------------------
# SAMPLE RETRIEVED CHUNKS
# ---------------------------------------------------------
#
# Normally ye chunks humare embedding/retrieval system
# se automatically aayenge.
#
# Abhi learning ke purpose se manually bana rahe hain.
# ---------------------------------------------------------

retrieved_chunks = [

    {
        "text": "Tuple Python ka ek collection/data structure hai. Isme hum multiple values ko ek saath store kar sakte hain.",
        "page": 1,
        "score": 0.82
    },

    {
        "text": "Tuple immutable hota hai. Existing elements ko change, add ya remove nahi kar sakte.",
        "page": 2,
        "score": 0.76
    },

    {
        "text": "Tuple indexing aur slicing support karta hai.",
        "page": 3,
        "score": 0.71
    }

]


# ---------------------------------------------------------
# CONTEXT BUILDER FUNCTION
# ---------------------------------------------------------
#
# Ye function retrieved chunks ko ek single textual
# context mein combine karega.
#
# LLM ko hum individual Python dictionaries nahi denge.
# Hum unka clean textual context banayenge.
# ---------------------------------------------------------

def build_context(retrieved_chunks):

    # Yahan har chunk ka formatted text store hoga.
    context_parts = []


    # Retrieved chunks ko one-by-one process kar rahe hain.
    for chunk in retrieved_chunks:

        # Source page ko context mein include kar rahe hain.
        #
        # Ye future mein citation/source display ke
        # liye useful hai.
        formatted_chunk = (
            f"[Source: Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

        # Formatted chunk ko list mein add kar rahe hain.
        context_parts.append(formatted_chunk)


    # -----------------------------------------------------
    # SAARE CHUNKS KO COMBINE KARNA
    # -----------------------------------------------------
    #
    # "\n\n---\n\n" separator hai.
    #
    # Isse LLM ko clearly pata chalega ki
    # ek chunk kahaan khatam hua aur doosra kahaan start hua.
    # -----------------------------------------------------

    context = "\n\n---\n\n".join(context_parts)


    # Final context return kar rahe hain.
    return context


# ---------------------------------------------------------
# BUILD CONTEXT
# ---------------------------------------------------------

context = build_context(retrieved_chunks)


# ---------------------------------------------------------
# DISPLAY GENERATED CONTEXT
# ---------------------------------------------------------

print("\n==============================")
print("GENERATED CONTEXT")
print("==============================\n")

print(context)


# ---------------------------------------------------------
# STEP 2: GROUNDED PROMPT
# ---------------------------------------------------------
#
# Is function ka kaam:
#
# User ka question + retrieved context
# ko combine karke ek grounded prompt banana.
#
# LLM ko hum clearly instructions denge:
#
# 1. Sirf provided context use karo.
# 2. Information invent mat karo.
# 3. Evidence insufficient ho to clearly batao.
# ---------------------------------------------------------

def build_grounded_prompt(question, context):

    prompt = f"""
You are DecisionIQ, an evidence-grounded AI assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer
the question, clearly say that there is not enough evidence
in the uploaded document.

Do not invent or assume information that is not present
in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


# ---------------------------------------------------------
# STEP 2.1: CREATE QUESTION
# ---------------------------------------------------------

question = "What is machine learning?"


# ---------------------------------------------------------
# STEP 2.2: BUILD GROUNDED PROMPT
# ---------------------------------------------------------

prompt = build_grounded_prompt(
    question,
    context
)


# ---------------------------------------------------------
# STEP 2.3: DISPLAY GROUNDED PROMPT
# ---------------------------------------------------------

print("\n==============================")
print("GROUNDED PROMPT")
print("==============================\n")

print(prompt)


# ---------------------------------------------------------
# STEP 4A: CONNECT GROUNDED PROMPT TO GEMINI
# ---------------------------------------------------------
#
# Ab hum actual Gemini client create karenge.
#
# Client automatically GEMINI_API_KEY environment
# variable se API key read karega.
#
# Isliye API key ko code mein directly nahi likh rahe.
# ---------------------------------------------------------

client = genai.Client()


# ---------------------------------------------------------
# SEND GROUNDED PROMPT TO GEMINI
# ---------------------------------------------------------
#
# Important:
#
# Hum Gemini ko sirf question nahi bhej rahe.
#
# Hum poora grounded prompt bhej rahe hain:
#
# Instructions
#      +
# Context
#      +
# Question
#
# Isse Gemini answer ko provided evidence ke saath
# ground kar sakta hai.
# ---------------------------------------------------------

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


# ---------------------------------------------------------
# DISPLAY FINAL GEMINI ANSWER
# ---------------------------------------------------------

print("\n==============================")
print("GROUNDED GEMINI ANSWER")
print("==============================\n")

print(response.text)