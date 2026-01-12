"""
Nahawi Web Editor - FastAPI Backend

A REST API for Arabic Grammar Correction using the NahawiEnsemble.

Usage:
    uvicorn main:app --reload --port 8000

API Documentation:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.routes import router

# Create FastAPI app
app = FastAPI(
    title="Nahawi - Arabic Grammar Correction API",
    description="""
## Nahawi (نحوي) - Arabic Grammatical Error Correction

Nahawi is a comprehensive Arabic GEC system achieving **78.84% F0.5** on QALB-2014 (3.79 points from SOTA).

### Features
- Correct 18 types of Arabic grammar errors
- Multiple correction strategies (cascading, parallel, specialist)
- Detailed error information with confidence scores
- Model contribution tracking

### Error Types Supported
- **Orthography**: hamza, taa_marbuta, alif_maqsura
- **Spelling**: letter confusions (د/ذ, ض/ظ, س/ص, ت/ط)
- **Morphology**: gender agreement, number agreement
- **Syntax**: missing/wrong prepositions
- **Verb**: conjugation errors
- **Article**: definiteness errors

### Quick Start
```python
import requests

response = requests.post(
    "http://localhost:8000/api/correct",
    json={"text": "اعلنت الحكومه عن خطه جديده"}
)
print(response.json()["corrected"])
# Output: أعلنت الحكومة عن خطة جديدة
```
    """,
    version="1.0.0",
    contact={
        "name": "Nahawi Project",
        "url": "https://github.com/nahawi"
    },
    license_info={
        "name": "MIT",
    }
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["correction"])


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nahawi Arabic Grammar Correction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "correct": "POST /api/correct",
            "status": "GET /api/status",
            "error_types": "GET /api/error-types"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
