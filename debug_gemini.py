"""
Diagnostic only - makes ONE raw Gemini call with no error handling, so whatever
is actually going wrong prints in full instead of being silently retried/hidden.

Run: python debug_gemini.py
"""
import os
import google.generativeai as genai

print(f"GEMINI_API_KEY set: {'yes' if os.environ.get('GEMINI_API_KEY') else 'NO - not set!'}")
print(f"Key starts with: {os.environ.get('GEMINI_API_KEY', '')[:8]}...")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("\nAvailable models that support generateContent:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(" -", m.name)

import config as C

print("\nTrying a real call with", C.GEMINI_MODEL, "...")
model = genai.GenerativeModel(C.GEMINI_MODEL)
resp = model.generate_content('Return strict JSON only: {"hello": "world"}')
print("RAW RESPONSE TEXT:")
print(repr(resp.text))
