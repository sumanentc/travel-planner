from deepagents import create_deep_agent, CompiledSubAgent
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt
from datetime import datetime
from openai import APITimeoutError, APIError
import httpx
import logging

from agents.prompts import SYSTEM_PROMPT_LOCAL_AGENT, SYSTEM_PROMPT_FOR_ITINERARY_AGENT
from models.agent_models import TravelAgentState, ItineraryDetails
from tools.custom_tools import internet_search
from utils import get_openai_model

logger = logging.getLogger(__name__)

local_agent = create_agent(
    model=get_openai_model("gpt-4o-mini"),
    tools=[internet_search],
    system_prompt=SYSTEM_PROMPT_LOCAL_AGENT
)

# Use it as a custom subagent
local_subagent = CompiledSubAgent(
    name="local_agent",
    description="A helpful agent that can suggest local activities or nearby places to visit.",
    runnable=local_agent
)
subagents = [local_subagent]


async def itinerary_planner(state: TravelAgentState, config: RunnableConfig):
    print(f'Inside itinerary-agent: {state}')
    # Get user query and conversation history
    user_query = state.get("regenerated_question") or state.get("user_query", "")
    messages = state.get("messages", [])
    
    # Validate that we have a query to process
    if not user_query:
        raise ValueError("No user query or regenerated question found in state")
    
    # Prepare messages: include conversation history only
    agent_messages = messages.copy() if messages else []
    
    # Get current date and other context
    current_date = datetime.now().strftime("%Y-%m-%d")
    trip_type = state.get("trip_type", "unknown")
    trip_purpose = state.get("trip_purpose", "unknown")
    responses = state.get("responses", {})
    
    # Extract transport and accommodation details from responses dict
    transport_details = responses.get("transport_details", "Not yet selected")
    accommodation_details = responses.get("accommodation_details", "Not yet selected")
    
    # Format responses as a string for the prompt
    responses_text = f"""
Transport Details: {transport_details}

Accommodation Details: {accommodation_details}
""".strip()
    
    # Format system prompt with all required fields including user_query
    full_system_prompt = SYSTEM_PROMPT_FOR_ITINERARY_AGENT.format(
        current_date=current_date,
        trip_type=trip_type,
        trip_purpose=trip_purpose,
        responses=responses_text,
        user_query=user_query
    )
    
    # Create deep agent with formatted prompt
    deep_agent = create_deep_agent(
        backend=FilesystemBackend(root_dir="/Users/sumandas/Public/workspace/python/roamwise_v2/data/", virtual_mode=True),
        model=get_openai_model("gpt-4o-mini"),
        system_prompt=full_system_prompt,
        tools=[internet_search],
        subagents=subagents,
        response_format=ProviderStrategy(ItineraryDetails),
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="internet_search",
                run_limit=5,
            )
        ],
    )
    
    # Invoke deep agent with messages
    itinerary_result = None
    try:
        result = await deep_agent.ainvoke(
            {"messages": agent_messages},
            config=config
        )
        
        if isinstance(result, dict) and "structured_response" in result:
            itinerary_result = result.get("structured_response")
        else:
            itinerary_result = result
        
        if itinerary_result is None:
            raise ValueError("No structured_response found in deep agent response")
        
        if not isinstance(itinerary_result, ItineraryDetails):
            raise ValueError(f"Expected ItineraryDetails, got {type(itinerary_result)}")
    except (APITimeoutError, httpx.ReadTimeout, httpx.TimeoutException) as e:
        logger.error(f"Timeout error in itinerary planner: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but the request timed out while creating your itinerary. "
            "This can happen when generating detailed travel plans takes longer than expected.\n\n"
            "Please try again with your request, or you can:\n"
            "• Request a simpler itinerary\n"
            "• Focus on specific activities\n"
            "• Rephrase your request\n\n"
            "Would you like to try again?"
        )
        agent_messages.append(AIMessage(content=error_message))
        user_input = interrupt(error_message)
        agent_messages.append(HumanMessage(content=user_input))
        return Command(
            update={
                "messages": agent_messages,
                "user_feedback": user_input,
            },
            goto="itinerary_agent"
        )
    except (APIError, Exception) as e:
        logger.error(f"Error in itinerary planner: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but I encountered an error while creating your itinerary. "
            "This might be a temporary issue.\n\n"
            "Please try again with your request, or you can:\n"
            "• Rephrase your request\n"
            "• Try a different approach\n\n"
            "Would you like to try again?"
        )
        agent_messages.append(AIMessage(content=error_message))
        user_input = interrupt(error_message)
        agent_messages.append(HumanMessage(content=user_input))
        return Command(
            update={
                "messages": agent_messages,
                "user_feedback": user_input,
            },
            goto="itinerary_agent"
        )
    
    # Handle case when user input is required (asking for confirmation or clarification)
    # Handle case when user input is required (multiple transport options)
    if itinerary_result.reply_to_user:
        # Validate that reply_to_user is not empty
        if not itinerary_result.reply_to_user.strip():
            raise ValueError("reply_to_user is present but empty")

        # Append AI message to conversation history before interrupt
        agent_messages.append(AIMessage(content=itinerary_result.reply_to_user))

        # Interrupt and wait for user response
        user_input = interrupt(itinerary_result.reply_to_user)

        # When graph resumes with Command(resume=value), execution continues HERE
        # The resume value is passed to user_input variable
        # Append user's response to conversation history
        agent_messages.append(HumanMessage(content=user_input))

        # Loop back to itinerary_agent to process the user's response
        return Command(
            update={
                "messages": agent_messages,
                "user_feedback": user_input,
            },
            goto="itinerary_agent"
        )
    
    # Return final result and populate current_state from next_step
    # Merge with existing responses dict
    existing_responses = state.get("responses", {})
    updated_responses = {**existing_responses, "itinerary_details": itinerary_result.recommended_itinerary_details}
    
    return Command(
        update={
            "reply_to_user": "",
            "responses": updated_responses,
            "current_state": 'itinerary',
            "messages": agent_messages,
            "user_feedback": ""
        }
    )