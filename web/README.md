# Nahawi Web Editor

A web-based Arabic Grammar Correction editor powered by the Nahawi ensemble model.

## Features

- Real-time Arabic grammar correction
- Error highlighting with color-coded categories
- Side-by-side before/after comparison
- Detailed error information with confidence scores
- Support for 13 error types

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 3. Start the Backend Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API will be available at http://localhost:8000

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

### 4. Start the Frontend Dev Server

In a new terminal:

```bash
cd frontend
npm run dev
```

The web editor will be available at http://localhost:5173

## One-Click Start (Windows)

```bash
run.bat
```

## One-Click Start (Linux/Mac)

```bash
./run.sh
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/correct` | Correct Arabic text |
| GET | `/api/status` | Get model status |
| GET | `/api/error-types` | Get supported error types |
| GET | `/api/health` | Health check |

### Example API Request

```bash
curl -X POST http://localhost:8000/api/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "اعلنت الحكومه عن خطه جديده"}'
```

### Example Response

```json
{
  "original": "اعلنت الحكومه عن خطه جديده",
  "corrected": "أعلنت الحكومة عن خطة جديدة",
  "corrections": [
    {
      "original": "اعلنت",
      "corrected": "أعلنت",
      "start": 0,
      "end": 5,
      "error_type": "hamza",
      "confidence": 0.95,
      "model": "hamza_fixer_rule_based"
    }
  ],
  "model_contributions": {
    "hamza_fixer_rule_based": 1,
    "taa_marbuta_fixer": 3
  },
  "confidence": 0.96,
  "processing_time_ms": 45.2
}
```

## Error Categories

| Category | Color | Error Types |
|----------|-------|-------------|
| Orthography | Red | hamza, taa_marbuta, alif_maqsura |
| Spelling | Orange | letter confusions (د/ذ, ض/ظ) |
| Morphology | Blue | gender/number agreement |
| Syntax | Purple | preposition errors |
| Verb | Cyan | conjugation errors |
| Article | Green | definiteness errors |

## Development

### Backend Development

The backend uses FastAPI with automatic API documentation.

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend Development

The frontend uses React + Vite + Tailwind CSS.

```bash
cd frontend
npm run dev
```

### Build for Production

```bash
cd frontend
npm run build
```

## Project Structure

```
web/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app
│   │   ├── components/
│   │   │   ├── Editor.jsx
│   │   │   ├── CorrectionPanel.jsx
│   │   │   ├── ErrorDetails.jsx
│   │   │   └── Header.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── README.md
├── run.bat                  # Windows startup script
└── run.sh                   # Linux/Mac startup script
```

## Performance

- Average response time: <100ms for typical sentences
- Supports texts up to 10,000 characters
- Lazy model loading for efficient memory usage

## License

MIT License - See main project LICENSE for details.
