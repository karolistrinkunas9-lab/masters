"""Run the simplified AI service"""
import uvicorn
# Import the custom head class so it's available for pickle deserialization
from app_simple import WeightedBinaryRelevanceHead  # noqa: F401

if __name__ == "__main__":
    uvicorn.run(
        "app_simple:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

