"""
Main application entry point for the stock prediction system.
"""

from ml_backend.api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 