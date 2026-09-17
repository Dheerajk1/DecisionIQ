# ---------------------------------------------------------
# DecisionIQ - First Gemini API Test
# ---------------------------------------------------------
#
# Purpose:
# Ab hum check karenge ki:
#
# Python
#    ↓
# google-genai SDK
#    ↓
# Gemini API
#    ↓
# Gemini Model
#    ↓
# Response
#
# properly kaam kar raha hai ya nahi.
# ---------------------------------------------------------

from google import genai


# ---------------------------------------------------------
# CREATE GEMINI CLIENT
# ---------------------------------------------------------
#
# Client automatically GEMINI_API_KEY
# environment variable se API key read karega.
#
# Isliye API key ko code mein directly nahi likh rahe.
# ---------------------------------------------------------

client = genai.Client()


# ---------------------------------------------------------
# SEND REQUEST TO GEMINI
# ---------------------------------------------------------

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain what a Python tuple is in simple words."
)


# ---------------------------------------------------------
# DISPLAY RESPONSE
# ---------------------------------------------------------

print("\n==============================")
print("GEMINI RESPONSE")
print("==============================\n")

print(response.text)