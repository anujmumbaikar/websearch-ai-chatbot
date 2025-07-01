from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict,Annotated,Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
import json
load_dotenv()