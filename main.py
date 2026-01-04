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
from google import genai
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

app = FastAPI()

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. DEFINE DATA MODELS (For FastAPI Docs & Validation)
class AnalysisData(BaseModel):
    extracted_skills: List[str]
    missing_skills: List[str]
    readiness_score: int
    recommended_resources: List[str]
    project_ideas: List[str]
    roadmap: List[str]

# 2. INITIALIZE CLIENTS
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

MONGO_URI = os.getenv("MONGO_URI")
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client.skillgap_db
collection = db.analysis_results

@app.get("/")
def read_root():
    return {"status": "online", "message": "SkillGap AI Server is Running!"}

@app.post("/analyze-skills")
async def analyze_skills(
    target_role: str = Form(...),
    resume_file: UploadFile = File(...)
):
    try:
        # 3. EXTRACT TEXT FROM PDF
        pdf_content = await resume_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

        # 4. IMPROVED AI PROMPT (Forcing content generation)
        prompt = f"""
        Analyze this resume for the role: '{target_role}'.
        Resume Content: {extracted_text}
        
        Return ONLY a JSON object with these EXACT keys:
        - "extracted_skills": list of skills found
        - "missing_skills": list of detailed at least 3 skills needed for this role
        - "readiness_score": integer 0-100
        - "recommended_resources": list of 4 specific basic Courses and tutorials in india
        - "project_ideas": list of detailed 3 practical projects to build
        - "roadmap": list of 3 clear steps to get hired
        
        Ensure no lists are empty.
        """

        # 5. CALL GEMINI 2.0 FLASH
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        analysis_data = json.loads(response.text)

        # 6. SAVE TO DATABASE
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
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}
# 1. NEW: Count of Queries Per Day
@app.get("/queries-per-day")
async def get_queries_per_day():
    # We fetch only the timestamps from MongoDB
    cursor = collection.find({}, {"timestamp": 1, "_id": 0})
    data = await cursor.to_list(length=1000)
    
    # Extract the date part (YYYY-MM-DD) from each timestamp
    dates = [d["timestamp"].strftime("%Y-%m-%d") for d in data]
    
    # Count how many times each date appears
    counts = Counter(dates)
    
    # Format for the Line/Bar Chart: [{"date": "2025-01-01", "count": 5}, ...]
    formatted = [{"date": k, "count": v} for k, v in sorted(counts.items())]
    return formatted

# 2. Top Missing Skills (Remains but verified)
@app.get("/top-missing-skills")
async def get_missing_skills():
    # Pull missing_skills from all documents
    cursor = collection.find({}, {"analysis.missing_skills": 1, "_id": 0})
    data = await cursor.to_list(length=100)
    
    # Flatten the list and count occurrences
    all_missing = [skill for doc in data for skill in doc["analysis"]["missing_skills"]]
    counts = Counter(all_missing).most_common(5)
    
    return [{"skill": k, "count": v} for k, v in counts]








if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
