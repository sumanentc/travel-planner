import logging
from datetime import datetime

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, SummarizationMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt
from openai import APITimeoutError, APIError

from agents.prompts import SYSTEM_PROMPT_ACCOMODATION_AGENT
from models.agent_models import TravelAgentState, AccommodationDetails
from tools.custom_tools import accommodation_search, calculator
from utils import get_openai_model

logger = logging.getLogger(__name__)


async def get_accomodation_details(state: TravelAgentState, config: RunnableConfig):
    """
    Accommodation agent that handles accommodation search and recommendations for travel planning.
    
    This agent searches for and recommends accommodation options based on user budget, preferences,
    destination, and travel dates. It calculates per-day accommodation costs, presents multiple 
    options to the user, handles feedback, and refines suggestions until the user confirms their choice.
    
    Workflow:
    1. Extract user query and conversation history from state
    2. Calculate per-day accommodation budget from total budget
    3. Create accommodation agent with search tools and structured output
    4. Search for accommodations based on cost, preferences, destination, and dates
    5. Present 3-4 options to user and wait for selection
    6. Handle user feedback and refine options if needed
    7. Extract complete information for selected option
    8. Ask user if they want to proceed with itinerary creation
    9. Once user confirms, save accommodation details and proceed to planning
    
    Args:
        state (TravelAgentState): Current state containing:
            - user_feedback: User's response to previous agent message (if any)
            - regenerated_question: Reformulated user query (if available)
            - user_query: Original user query
            - messages: Conversation history
            - trip_purpose: Purpose of trip (leisure, business, adventure, etc.)
        config (RunnableConfig): Runtime configuration for the agent
    
    Returns:
        Command: LangGraph command with updated state:
            - If user input required: Returns with interrupt and loops back to accommodation_agent
            - If accommodation confirmed: Updates responses and proceeds to next step
    
    Raises:
        ValueError: If no user query is found in state
        APITimeoutError: If accommodation search times out (handled with retry)
        APIError: If API call fails (handled with retry)
    
    State Updates:
        - responses: Accommodation details formatted for response aggregation
        - messages: Updated conversation history
        - user_feedback: User's latest input
        - current_state: Next step in workflow (e.g., "planning")
        - reply_to_user: Message to display to user (when presenting options)
    """
    # Check for user_feedback first (in case of interrupt resume), then regenerated_question, then user_query
    user_query = state.get("user_feedback", "") or state.get("regenerated_question") or state.get("user_query", "")
    messages = state.get("messages", [])
    trip_purpose = state.get("trip_purpose", "unknown")

    # Validate that we have a query to process
    if not user_query:
        raise ValueError("No user query, feedback, or regenerated question found in state")
    print(f'Inside accomodation-agent: {state}')

    # Prepare messages: include conversation history only
    agent_messages = messages.copy() if messages else []

    # Get current date for prompt formatting
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Format system prompt with trip_purpose and user_query
    formatted_system_prompt = SYSTEM_PROMPT_ACCOMODATION_AGENT.format(
        trip_purpose=trip_purpose,
        current_date=current_date,
        user_query=user_query
    )

    # Log retry configuration
    logger.info(f"Accommodation agent - Model config: timeout=60s, max_retries=2")
    logger.info(f"Accommodation agent - Message count: {len(agent_messages)}")

    accommodation_agent = create_agent(
        model=get_openai_model("gpt-4o-mini"),
        tools=[calculator,accommodation_search],
        system_prompt=formatted_system_prompt,
        response_format=ProviderStrategy(AccommodationDetails),
        middleware=[
            SummarizationMiddleware(
                model=get_openai_model("gpt-4o-mini"),
                max_tokens_before_summary=8000,  # Increased from 4000 to handle longer search results
                messages_to_keep=20,
            ),
            ToolCallLimitMiddleware(
                tool_name="accommodation_search",
                run_limit=5,  # Limit accommodation_search as it's more expensive (GPTResearcher)
            )
        ],
    )


    # Invoke accommodation_agent with full message history
    accommodation_details = None
    try:
        logger.info("Accommodation agent - Starting agent invocation...")

        result = await accommodation_agent.ainvoke(
            {"messages": agent_messages},
            config=config
        )

        if isinstance(result, dict) and "structured_response" in result:
            accommodation_details = result.get("structured_response")
        else:
            accommodation_details = result

        if accommodation_details is None:
            raise ValueError(
                f"Missing 'structured_response' in accommodation response."
            )

        if not isinstance(accommodation_details, AccommodationDetails):
            raise ValueError(
                f"Invalid response type from accommodation agent. "
                f"Expected AccommodationDetails, got {type(accommodation_details).__name__}"
            )
        
        logger.info(f"Accommodation agent - Successfully processed. Reply to user: {bool(accommodation_details.reply_to_user)}")
    except (APITimeoutError, httpx.ReadTimeout, httpx.TimeoutException) as e:
        logger.error(f"Timeout error in accommodation agent: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but the request timed out while searching for accommodations. "
            "This can happen when the search takes longer than expected.\n\n"
            "Please try again with your request, or you can:\n"
            "• Simplify your search criteria\n"
            "• Try a different destination\n"
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
            goto="accommodation_agent"
        )
    except (APIError, Exception) as e:
        logger.error(f"Error in accommodation agent: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but I encountered an error while searching for accommodations. "
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
            goto="accommodation_agent"
        )

    # Handle case when user input is required (multiple accommodation options)
    if accommodation_details.reply_to_user:
        # Validate that reply_to_user is not empty
        if not accommodation_details.reply_to_user.strip():
            raise ValueError("reply_to_user is present but empty")

        # Append AI message to conversation history before interrupt
        agent_messages.append(AIMessage(content=accommodation_details.reply_to_user))

        # Interrupt and wait for user response
        user_input = interrupt(accommodation_details.reply_to_user)

        # When graph resumes with Command(resume=value), execution continues HERE
        # The resume value is passed to user_input variable
        # Append user's response to conversation history
        agent_messages.append(HumanMessage(content=user_input))

        # Loop back to accommodation_agent to process the user's response
        return Command(
            update={
                "messages": agent_messages,
                "user_feedback": user_input,
            },
            goto="accommodation_agent"
        )

    # Return final result for other inquiry types
    # Merge with existing responses dict
    existing_responses = state.get("responses", {})
    updated_responses = {**existing_responses, "accommodation_details": accommodation_details.recommended_accommodation_details}
    
    return Command(
        update={
            "reply_to_user": "",
            "responses": updated_responses,
            "messages": agent_messages,
            "user_feedback": "",
            "current_state": "accommodation"
        }
    )