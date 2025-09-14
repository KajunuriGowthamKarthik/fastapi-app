from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests
import json
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("post_generator_api")

# --- App Initialization ---
app = FastAPI(
    title=" LinkedIn Post Generator API",
    description="AI-powered LinkedIn post generator with analytics, templates, and multi-model support.",
    version="1.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify trusted origins!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security (Bearer Token) ---
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    token = credentials.credentials
    if token != "test-api-key":  # Set your token here
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# --- Pydantic Models ---
class PostRequest(BaseModel):
    topic: str = Field(..., example="Write a LinkedIn post about the future of AI in business.")
    template: Optional[str] = Field(None, example="tips|story|announcement")
    hashtags: Optional[List[str]] = Field(None, example=["#AI", "#Business"])
    author: Optional[Dict[str, str]] = Field(
        None, 
        example={"name": "Karthik", "title": "CEO at WildCoder"}
    )
    model: Optional[str] = Field("llama-3.3-70b-versatile", description="AI model to use")

class PostResponse(BaseModel):
    post: str
    meta: Optional[Dict] = None

# --- Supported Model Choices ---
MODEL_LIST = [
    "llama-3.3-70b-versatile",
    "llama-2-70b",
    "mixtral-8x22b",
    "gpt-4o",
    "gpt-3.5-turbo"
]

AI_SYSTEM_PROMPT = (
    "You are a The best helpful assistant and expert in professional LinkedIn social content writing. and use emojies if needed"
    " for every for maintain a proper title, intro then main content of the post then ending and last relevant #tags Return only the content for the post, without explanation." 
)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"  # Your full valid Groq API key

@app.post("/post_generator/", response_model=PostResponse)
@limiter.limit("10/minute")  # Per-IP!
async def post_generator(
    request: Request,        # Needed for SlowAPI!
    body: PostRequest,
    token: str = Depends(verify_token)
):
    """
    Generate AI-powered LinkedIn post content with optional templates, hashtags, and author metadata.
    """
    logger.info(f"New post generation - topic: {body.topic[:40]}...")

    # Validate model selection
    model_choice = body.model or "llama-3.3-70b-versatile"
    if model_choice not in MODEL_LIST:
        raise HTTPException(status_code=400, detail="Unsupported model.")

    # Construct prompt
    prompt_parts = [body.topic]
    if body.template:
        prompt_parts.append(
            f"Use the '{body.template}' LinkedIn post template. Structure the post accordingly."
        )
    if body.hashtags:
        prompt_parts.append("Include these hashtags at the end: " + " ".join(body.hashtags))
    if body.author:
        prompt_parts.append(
            f"Write as {body.author.get('name', 'A professional')} ({body.author.get('title', '')})."
        )
    prompt = "\n".join([AI_SYSTEM_PROMPT] + prompt_parts)

    ai_payload = {
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "model": model_choice
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    # Communicate with Groq API
    try:
        ai_response = requests.post(
            GROQ_CHAT_URL, headers=headers, json=ai_payload, timeout=30
        )
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=502, detail="Error contacting AI engine.")

    if ai_response.status_code != 200:
        logger.error(f"AI error: {ai_response.text}")
        raise HTTPException(status_code=502, detail="AI engine error")

    try:
        ai_data = ai_response.json()
        content = ai_data.get('choices',[{}])[0].get('message',{}).get('content')
        if not content:
            raise KeyError
    except Exception as e:
        logger.error(f"Response parsing error: {e}")
        raise HTTPException(status_code=500, detail="Malformed AI response")

    meta = {
        "model": model_choice,
        "chars": len(content),
        "request_id": ai_data.get("id"),
        "usage": ai_data.get("usage"),
    }

    logger.info(f"Post generated successfully ({len(content)} chars)")
    return PostResponse(
        post=content.strip(),
        meta=meta
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model_list")
def model_list():
    return {"models": MODEL_LIST}
