import uvicorn
from pathlib import Path

if __name__ == "__main__":
    uvicorn.run("src.service.api:app", host="127.0.0.1", port=8000, reload=False)
