from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from langchain.chat_models import init_chat_model 
from langchain_tavily import TavilySearch
from langchain_community.tools import BraveSearch
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage,AIMessageChunk
from langgraph.prebuilt import ToolNode,tools_condition
import json
from langgraph.graph.message import add_messages
import requests
import asyncio
from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool()
def get_weather(city: str):
    """This tool returns the weather data about the given city"""
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Error fetching weather data for {city}"
    return "The weather in " + city + " is " + response.text

@tool
def search(query: str):
    """This tool searches the web for the given query"""
    search = TavilySearch(max_results=5)
    results = search.invoke(query)
    return "Results:" + json.dumps(results, indent=2)

tools=[get_weather,search] 
llm = init_chat_model( model_provider="openai", model="gpt-4.1-mini")
llm_with_tools = llm.bind_tools(tools)

async def chat_node(state: State):
    message = await llm_with_tools.ainvoke(state["messages"])
    return {
        "messages": [message]
    }

async def tools_router(state: State):
    last_message = state["messages"][-1]
    if(hasattr(last_message,"tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return END

# tool_node = ToolNode(tools=tools)
# there is inbuit tool node in langgrah but , just to streaming purpose we are using our own tool node
# internally its same as ToolNode
async def tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # Choose the tool and invoke
        if tool_name == "search":
            result = await search.ainvoke(tool_args)
        elif tool_name == "get_weather":
            result = await get_weather.ainvoke(tool_args)
        else:
            result = f"Tool '{tool_name}' not implemented."

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_id,
            tool_name=tool_name,
        )
        tool_messages.append(tool_message)

    return {
        "messages": tool_messages
    }



memory = MemorySaver()
graph_builder = StateGraph(State)
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_node("tools",tool_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_conditional_edges("chat_node",tools_condition)
graph_builder.add_edge("tools", "chat_node")
graph = graph_builder.compile(checkpointer=memory)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

def serialize_ai_message_chunk(chunk):
    if(isinstance(chunk,AIMessageChunk)):
        return chunk.content
    else:
        raise TypeError(
            f"Expected AIMessageChunk, got {type(chunk)}"
        )
async def generate_chat_response(message: str, checkpoint_id: Optional[str] = Query(None)):
    is_new_conversation = checkpoint_id is None
    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config =  {
            "configurable":{
                "thread_id": new_checkpoint_id
            }
        }
        _state = {
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        events = graph.astream_events(
            _state,
            config=config,
            version="v2"
        )
        yield f"data:{{\"type\":\"checkpoint_id\",\"data\":\"{new_checkpoint_id}\"}}\n\n"
    
    else:
        config =  {
            "configurable":{
                "thread_id": checkpoint_id
            }
        }
        _state = {
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        events = graph.astream_events(
            _state,
            config=config,
            version="v2"
        )

    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialize_ai_message_chunk(event["data"]["chunk"])
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")
            yield f"data:{{\"type\":\"chunk\",\"data\":\"{safe_content}\"}}\n\n"
        
        elif event_type == "on_chat_model_end":
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "search"]
            weather_calls = [call for call in tool_calls if call["name"] == "get_weather"]

            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = search_query.replace("'", "\\'").replace("\n", "\\n")
                yield f"data:{{\"type\":\"search\",\"data\":\"{safe_query}\"}}\n\n"
            if weather_calls:
                weather_city = weather_calls[0]["args"].get("city", "")
                safe_city = weather_city.replace("'", "\\'").replace("\n", "\\n")
                yield f"data:{{\"type\":\"weather\",\"data\":\"{safe_city}\"}}\n\n"
        
        elif event_type == "on_tool_end" and event["name"] == "search":
            output = event["data"]["output"]
            if isinstance(output, list):
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])
                urls_json = json.dumps(urls)
                yield f"data:{{\"type\":\"search_results\",\"data\":{urls_json}}}\n\n"
        
        
    yield f"data:{{\"type\":\"end\"}}\n\n"


@app.get("/chat_stream/{message}")
async def chat_stream(message: str,checkpoint_id:Optional[str]=Query(None)):
    #SSE - Server-Sent Events
    # why ? because we want to stream the response from the server to the client
    # we can hv used WebSockets,but it can be overkill for this use case
    # we want to stream the response from the server to the client
    return StreamingResponse(
        generate_chat_response(message,checkpoint_id),
        media_type="text/event-stream"
    )
