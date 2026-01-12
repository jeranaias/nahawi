"""Pydantic models for API request/response schemas."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class CorrectionRequest(BaseModel):
    """Request body for correction endpoint."""
    text: str = Field(..., min_length=1, max_length=10000, description="Arabic text to correct")
    strategy: str = Field(default="cascading", description="Correction strategy: cascading, parallel, or specialist")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence to apply correction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "اعلنت الحكومه عن خطه جديده لتطوير البنيه",
                    "strategy": "cascading",
                    "confidence_threshold": 0.7
                }
            ]
        }
    }


class CorrectionItem(BaseModel):
    """A single correction."""
    original: str = Field(..., description="Original text that was corrected")
    corrected: str = Field(..., description="Corrected text")
    start: int = Field(..., description="Start character position in original text")
    end: int = Field(..., description="End character position in original text")
    error_type: str = Field(..., description="Type of error: hamza, taa_marbuta, etc.")
    confidence: float = Field(..., description="Model confidence in this correction")
    model: str = Field(..., description="Model that made this correction")


class CorrectionResponse(BaseModel):
    """Response body for correction endpoint."""
    original: str = Field(..., description="Original input text")
    corrected: str = Field(..., description="Corrected output text")
    corrections: List[CorrectionItem] = Field(default_factory=list, description="List of corrections made")
    model_contributions: Dict[str, int] = Field(default_factory=dict, description="Count of corrections per model")
    confidence: float = Field(..., description="Overall confidence score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                    "model_contributions": {"hamza_fixer_rule_based": 1, "taa_marbuta_fixer": 3},
                    "confidence": 0.96,
                    "processing_time_ms": 45.2
                }
            ]
        }
    }


class ModelInfo(BaseModel):
    """Information about a single model."""
    name: str
    error_types: List[str]
    is_loaded: bool
    model_type: str


class StatusResponse(BaseModel):
    """Response body for status endpoint."""
    status: str = Field(..., description="Service status: ok, degraded, error")
    models: List[ModelInfo] = Field(default_factory=list, description="List of available models")
    total_models: int = Field(..., description="Total number of models")
    loaded_models: int = Field(..., description="Number of currently loaded models")


class ErrorTypeInfo(BaseModel):
    """Information about an error type."""
    name: str
    description: str
    category: str
    examples: List[str]


class ErrorTypesResponse(BaseModel):
    """Response body for error-types endpoint."""
    error_types: List[ErrorTypeInfo]
