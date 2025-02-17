from typing import Sequence

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."
language = "Silesian"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages, "language": language}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state


query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages, "language": language}, config)
output["messages"][-1].pretty_print()
