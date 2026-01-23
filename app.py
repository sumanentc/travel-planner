from typing import List, Dict, Any

import chainlit as cl
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from workflow import get_travel_agent

# Initialize in-memory checkpointer and store at application level
checkpointer = MemorySaver()
store = InMemoryStore()


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Plan a family trip to Manali",
            message="Plan a 5 day trip to Manali from Mumbai in the month of May",
        ),
        cl.Starter(
            label="Plan a weekend Gateway to Lonavala",
            message="Plan a weekend trip to Lonavala from Mumbai in the last week of Dec",
        ),
        cl.Starter(
            label="Where to go in the month May",
            message="Best places to visit in the month of may in india from Mumbai",
        )
    ]


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize a new chat session.
    This callback is triggered when a new conversation starts.
    Starters will be displayed automatically if no message is sent here.
    """
    # Store the thread_id in user session for reference
    thread_id = cl.context.session.id
    cl.user_session.set("thread_id", thread_id)


@cl.on_chat_resume
async def on_chat_resume(thread: Dict[str, Any]):
    """
    Resume an existing chat session from history.
    This callback is triggered when a user returns to a previous conversation.
    """
    # Restore the thread_id in user session
    thread_id = thread.get("id")
    cl.user_session.set("thread_id", thread_id)
    
    # Optional: Send a message indicating the chat has been resumed
    await cl.Message(
        content="💬 Chat resumed. You can continue from where you left off!"
    ).send()

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    print(config)
    final_answer = cl.Message(content="")
    graph = get_travel_agent(checkpointer, store)

    # Check if there's an interrupted state to resume
    state = await graph.aget_state(config)

    if state.next:  # Graph is interrupted and waiting for input
        # Get all pending tasks (interrupts)
        tasks = state.tasks
        
        # Find tasks with interrupts
        interrupted_tasks = [task for task in tasks if task.interrupts]
        
        # When there are multiple interrupts, we need to specify which one to resume
        # LangGraph requires us to provide a dict mapping task_id -> resume_value
        if len(interrupted_tasks) > 1:
            # Resume only the most recent interrupt (last one in the list)
            # This is the one the user is responding to
            latest_task = interrupted_tasks[-1]
            resume_value = {latest_task.id: msg.content}
            result = await graph.ainvoke(
                    Command(resume=resume_value),
                    config=RunnableConfig(**config)
            )
            # Handle interrupt messages
            if "__interrupt__" in result:
                for interrupt_item in result["__interrupt__"]:
                    if hasattr(interrupt_item, "value"):
                        await final_answer.stream_token(interrupt_item.value)
            # Handle regular replies
            elif "reply_to_user" in result and result["reply_to_user"]:
                await final_answer.stream_token(result["reply_to_user"])
        else:

            # Single interrupt - resume normally with just the message content
            result = await graph.ainvoke(
                    Command(resume=msg.content),
                    config=RunnableConfig(**config)
            )
            # Handle interrupt messages
            if "__interrupt__" in result:
                for interrupt_item in result["__interrupt__"]:
                    if hasattr(interrupt_item, "value"):
                        await final_answer.stream_token(interrupt_item.value)
            # Handle regular replies
            elif "reply_to_user" in result and result["reply_to_user"]:
                await final_answer.stream_token(result["reply_to_user"])
    else:
        # No interrupts - either truly new conversation or continuing after END
        # Preserve message history to detect follow-ups
        existing_messages = state.values.get("messages", []) if state.values else []
        
        result = await graph.ainvoke(
                {
                    "user_query": msg.content, 
                    "messages": existing_messages,
                    "reply_to_user": "",
                    "current_state": "",
                    "user_feedback": ""
                },
                config=RunnableConfig(**config)
        )
        # Handle interrupt messages
        if "__interrupt__" in result:
            for interrupt_item in result["__interrupt__"]:
                if hasattr(interrupt_item, "value"):
                    await final_answer.stream_token(interrupt_item.value)
        # Handle regular replies
        elif "reply_to_user" in result and result["reply_to_user"]:
            await final_answer.stream_token(result["reply_to_user"])

    await final_answer.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)