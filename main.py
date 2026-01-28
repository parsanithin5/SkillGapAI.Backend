import os
import io
import json
import datetime
from typing import List
from collections import Counter

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2

from motor.motor_asyncio import AsyncIOMotorClient
from mistralai import Mistral
import re

# --------------------------------------------------
# LOAD ENV VARIABLES
# --------------------------------------------------
load_dotenv()

def extract_json(text: str):
    """
    Safely extract JSON object from Mistral response
    """
    try:
        # Remove markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()

        # Extract first JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")

        return json.loads(match.group())
    except Exception as e:
        raise ValueError(f"Invalid JSON from Mistral: {e}")


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# DATA MODELS
# --------------------------------------------------
class AnalysisData(BaseModel):
    extracted_skills: List[str]
    missing_skills: List[str]
    readiness_score: int
    recommended_resources: List[str]
    project_ideas: List[str]
    roadmap: List[str]

# --------------------------------------------------
# INITIALIZE MISTRAL CLIENT
# --------------------------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# --------------------------------------------------
# MONGODB SETUP
# --------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client.skillgap_db
collection = db.analysis_results

# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "online", "message": "SkillGap AI Server is Running (Mistral)!"}

# --------------------------------------------------
# ANALYZE SKILLS ENDPOINT
# --------------------------------------------------
@app.post("/analyze-skills")
async def analyze_skills(
    target_role: str = Form(...),
    resume_file: UploadFile = File(...)
):
    try:
        # 1. EXTRACT TEXT FROM PDF
        pdf_content = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        extracted_text = ""

        for page in pdf_reader.pages:
            if page.extract_text():
                extracted_text += page.extract_text()

        # 2. PROMPT
        prompt = f"""
You are an AI career analyst.

Analyze the following resume for the role: "{target_role}"

Resume Content:
{extracted_text}

Return ONLY valid JSON with EXACT keys:
- extracted_skills (list)
- missing_skills (minimum 3)
- readiness_score (0 to 100)
- recommended_resources (4 beginner-friendly courses/tutorials in India)
- project_ideas (3 practical project ideas)
- roadmap (3 clear steps to get hired)

Do not add explanations or extra text.
"""

        # 3. CALL MISTRAL
        response = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        raw_output = response.choices[0].message.content
        analysis_data = extract_json(raw_output)

        # 4. SAVE TO DATABASE
        document = {
            "filename": resume_file.filename,
            "target_role": target_role,
            "analysis": analysis_data,
            "timestamp": datetime.datetime.utcnow()
        }

        await collection.insert_one(document)

        return {
            "status": "success",
            "data": analysis_data
        }

    except Exception as e:
        print("Error:", e)
        return {"status": "error", "message": str(e)}

# --------------------------------------------------
# ANALYTICS ENDPOINTS
# --------------------------------------------------

# 1. Queries Per Day
@app.get("/queries-per-day")
async def get_queries_per_day():
    cursor = collection.find({}, {"timestamp": 1, "_id": 0})
    data = await cursor.to_list(length=1000)

    dates = [d["timestamp"].strftime("%Y-%m-%d") for d in data]
    counts = Counter(dates)

    return [{"date": k, "count": v} for k, v in sorted(counts.items())]

# 2. Top Missing Skills
@app.get("/top-missing-skills")
async def get_missing_skills():
    cursor = collection.find({}, {"analysis.missing_skills": 1, "_id": 0})
    data = await cursor.to_list(length=100)

    all_missing = [
        skill
        for doc in data
        for skill in doc["analysis"]["missing_skills"]
    ]

    counts = Counter(all_missing).most_common(5)

    return [{"skill": k, "count": v} for k, v in counts]

# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )