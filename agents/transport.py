import logging
from datetime import datetime

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt
from openai import APITimeoutError, APIError

from agents.prompts import SYSTEM_PROMPT_TRANSPORT_AGENT
from models.agent_models import TravelAgentState, TransportDetails
from tools.custom_tools import transport_search, calculator
from utils import get_openai_model

logger = logging.getLogger(__name__)

async def get_transport_details(state: TravelAgentState, config: RunnableConfig):
    """
    Transport agent that handles transportation search and recommendations for travel planning.
    
    This agent searches for and recommends transportation options based on user preferences,
    budget, and trip details. It presents multiple options to the user, handles feedback,
    and refines suggestions until the user confirms their choice.
    
    Workflow:
    1. Extract user query and conversation history from state
    2. Create transport agent with search tools and structured output
    3. Search for transportation options (flights, trains, buses, car rentals, etc.)
    4. Present 3-4 options to user and wait for selection
    5. Handle user feedback and refine options if needed
    6. Once user confirms, save transport details and proceed to accommodation
    
    Args:
        state (TravelAgentState): Current state containing:
            - user_feedback: User's response to previous agent message (if any)
            - regenerated_question: Reformulated user query (if available)
            - messages: Conversation history
            - trip_type: Type of trip (domestic, international, weekend, etc.)
        config (RunnableConfig): Runtime configuration for the agent
    
    Returns:
        Command: LangGraph command with updated state:
            - If user input required: Returns with interrupt and loops back to transport_agent
            - If transport confirmed: Updates transport_details and proceeds to next step
    
    Raises:
        ValueError: If no user query is found in state
        APITimeoutError: If transport search times out (handled with retry)
        APIError: If API call fails (handled with retry)
    
    State Updates:
        - transport_details: Selected transport option details (when confirmed)
        - responses: Transport details formatted for response aggregation
        - messages: Updated conversation history
        - user_feedback: User's latest input
        - current_state: Next step in workflow (e.g., "accommodation")
        - reply_to_user: Message to display to user (when presenting options)
    """
    print(f'Inside transport-agent: {state}')
    # Check for user_feedback first (in case of interrupt resume), then regenerated_question, then user_query
    user_query = state.get("user_feedback", "") or state.get("regenerated_question")
    messages = state.get("messages", [])
    trip_type = state.get("trip_type", "unknown")
    
    # Validate that we have a query to process
    if not user_query:
        raise ValueError("No user query, feedback, or regenerated question found in state")

    # Prepare messages: include conversation history only
    agent_messages = messages.copy() if messages else []

    # Get current date for prompt formatting
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Format system prompt with trip_type and user_query
    formatted_system_prompt = SYSTEM_PROMPT_TRANSPORT_AGENT.format(
        trip_type=trip_type,
        current_date=current_date,
        user_query=user_query
    )
    
    # Log retry configuration
    logger.info(f"Transport agent - Model config: timeout=60s, max_retries=2")
    logger.info(f"Transport agent - Message count: {len(agent_messages)}")
    
    transport_agent = create_agent(
        model=get_openai_model("gpt-4o-mini"),
        tools=[calculator,transport_search],
        system_prompt=formatted_system_prompt,
        response_format=ProviderStrategy(TransportDetails),
        middleware=[
            SummarizationMiddleware(
                model=get_openai_model("gpt-4o-mini"),
                max_tokens_before_summary=8000,  # Increased from 4000 to handle longer search results
                messages_to_keep=20,
            ),
            ToolCallLimitMiddleware(
                tool_name="transport_search",
                run_limit=3,  # Limit transport_search as it's more expensive (GPTResearcher)
            )
        ],
    )
    
    # Invoke transport_agent with full message history
    transport_result = None
    try:
        logger.info("Transport agent - Starting agent invocation...")

        result = await transport_agent.ainvoke(
            {"messages": agent_messages},
            config=config
        )

        if isinstance(result, dict) and "structured_response" in result:
            transport_result = result.get("structured_response")
        else:
            transport_result = result

        if transport_result is None:
            raise ValueError(
                f"Missing 'structured_response' in transport response."
            )

        if not isinstance(transport_result, TransportDetails):
            raise ValueError(
                f"Invalid response type from validator agent. "
                f"Expected TransportDetails, got {type(transport_result).__name__}"
            )
        
        logger.info(f"Transport agent - Successfully processed. Reply to user: {bool(transport_result.reply_to_user)}")
    except (APITimeoutError, httpx.ReadTimeout, httpx.TimeoutException) as e:
        logger.error(f"Timeout error in transport agent: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but the request timed out while searching for transportation options. "
            "This can happen when the search takes longer than expected.\n\n"
            "Please try again with your request, or you can:\n"
            "• Simplify your search criteria\n"
            "• Try a different route\n"
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
            goto="transport_agent"
        )
    except (APIError, Exception) as e:
        logger.error(f"Error in transport agent: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but I encountered an error while searching for transportation options. "
            "This might be a temporary issue.\n\n"
            "Please try again with your request, or you can:\n"
            "• Rephrase your request\n"
            "• Try a different search\n\n"
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
            goto="transport_agent"
        )

    # Handle case when user input is required (multiple transport options)
    if transport_result.reply_to_user:
        # Validate that reply_to_user is not empty
        if not transport_result.reply_to_user.strip():
            raise ValueError("reply_to_user is present but empty")
        
        # Append AI message to conversation history before interrupt
        agent_messages.append(AIMessage(content=transport_result.reply_to_user))

        # Interrupt and wait for user response
        user_input = interrupt(transport_result.reply_to_user)

        # When graph resumes with Command(resume=value), execution continues HERE
        # The resume value is passed to user_input variable
        # Append user's response to conversation history
        agent_messages.append(HumanMessage(content=user_input))

        # Loop back to transport_agent to process the user's response
        return Command(
            update={
                "messages": agent_messages,
                "user_feedback": user_input,
            },
            goto="transport_agent"
        )

    # Return final result for other inquiry types
    # Merge with existing responses dict
    existing_responses = state.get("responses", {})
    updated_responses = {**existing_responses, "transport_details": transport_result.recommended_transport_details}
    
    return Command(
        update={
            "reply_to_user": "",
            "responses": updated_responses,
            "messages": agent_messages,
            "user_feedback": "",
            "current_state": "transport"
        }
    )