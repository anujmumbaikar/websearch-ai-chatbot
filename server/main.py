from dotenv import load_dotenv
from .app import app
import uvicorn

load_dotenv()

def main():
    uvicorn.run(app, port=8000, host="0.0.0.0")

main()