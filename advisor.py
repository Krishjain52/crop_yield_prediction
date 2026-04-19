import groq
import streamlit as st
import json
import re

client = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_farm_advisory(area, item, year, rainfall, pesticides, temperature, predicted_yield_kg):
    
    prompt = f"""
You are an expert agricultural advisor. A farmer has provided the following details:

- Country/Region: {area}
- Crop: {item}
- Year: {year}
- Average Rainfall: {rainfall} mm/year
- Pesticides Used: {pesticides} tonnes
- Average Temperature: {temperature} °C
- Predicted Yield: {predicted_yield_kg:.0f} kg/ha

Analyze these farm conditions and return ONLY a JSON object with exactly this structure.
Keep all text values short and simple. Do not use special characters or quotes inside text values.

{{
  "crop_summary": "2-3 sentence summary here",
  "yield_interpretation": "What this yield means for this crop and region",
  "risk_factors": [
    {{"factor": "Risk name", "severity": "High", "explanation": "Brief explanation"}},
    {{"factor": "Risk name", "severity": "Medium", "explanation": "Brief explanation"}},
    {{"factor": "Risk name", "severity": "Low", "explanation": "Brief explanation"}}
  ],
  "recommended_actions": [
    {{"action": "What to do", "reason": "Why this helps"}},
    {{"action": "What to do", "reason": "Why this helps"}},
    {{"action": "What to do", "reason": "Why this helps"}}
  ],
  "disclaimer": "This is AI-generated advice. Always consult a local agricultural expert before making farm decisions."
}}

Rules:
- Return ONLY the JSON object
- No text before or after the JSON
- No markdown or code fences
- No apostrophes or quotes inside text values
- Keep each text value under 100 words
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()

    # Remove code fences if present
    raw = re.sub(r"```json|```", "", raw).strip()

    # Find the JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    raw = raw[start:end]

    advisory = json.loads(raw)
    return advisory