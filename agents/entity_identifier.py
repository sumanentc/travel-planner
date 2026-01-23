import logging
from datetime import datetime

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from langgraph.constants import END
from openai import APITimeoutError, APIError

from agents.prompts import SYSTEM_PROMPT_ENTITY_IDENTIFIER
from models.agent_models import TravelInquiry, TravelAgentState
from tools.custom_tools import web_search, calculator
from utils import get_openai_model

logger = logging.getLogger(__name__)


async def collect_missing_entities(state: TravelAgentState, config: RunnableConfig):
    """
    Extracts and validates required entities from travel queries, ensuring all necessary 
    information is collected before proceeding to the next agent.
    
    This function:
    1. Extracts entities from the entire conversation history:
       - destination, origin_city, travel_dates, trip_duration, number_of_travelers, user_preferences, budget
    2. Asks for ONE missing entity at a time in priority order
    3. Validates budget sufficiency using web_search (if budget appears insufficient, asks for confirmation)
    4. Regenerates fully qualified standalone questions when all entities are collected
    5. Handles follow-up questions by resolving pronouns with explicit entities from conversation
    
    Entity Collection Flow:
    - If entities missing: Sets reply_to_user with question, interrupts for user input, loops back to entity_identifier
    - If all collected: Sets reply_to_user="", populates regenerated_question with complete query, proceeds to next step
    
    Args:
        state: TravelAgentState containing:
            - user_feedback: User's response to previous question (if resuming from interrupt)
            - regenerated_question: Travel query from validator (if first call)
            - messages: Full conversation history
        config: Runnable configuration for the agent
        
    Returns:
        Command object with state updates:
        - reply_to_user: Question for missing entity (if incomplete) or "" (if complete)
        - regenerated_question: Complete standalone question with ALL entities (if complete) or "" (if incomplete)
        - number_of_travelers: Extracted number of travelers or None
        - current_state: Next step identifier from inquiry_result.next_step
        - messages: Updated conversation history
        - user_feedback: User's input (when looping back) or "" (when complete)
        
    Routing:
        - goto="entity_identifier": When entities are missing (loops back after user provides input)
        - No goto (proceeds): When all entities collected and regenerated_question is populated
    """
    print(f'Inside entity-identifier: {state}')
    # Get user query, messages, and existing entities
    user_query = state.get("user_feedback", "") or state.get("regenerated_question", "")
    messages = state.get("messages", [])
    
    # Validate that we have a query to process
    if not user_query:
        raise ValueError("No user query or feedback found in state")

    # Prepare messages: include conversation history + current query
    agent_messages = messages.copy() if messages else []

    # Only append user_query if not resuming from interrupt (avoid duplicate messages)
    # When resuming, user_feedback is already in the message history
    if not state.get("user_feedback"):
        agent_messages.append(HumanMessage(content=user_query))

    # Get current date and query classification from state
    current_date = datetime.now().strftime("%Y-%m-%d")
    query_type = state.get("query_type", "trip_planning")
    trip_type = state.get("trip_type", "null")
    trip_purpose = state.get("trip_purpose", "null")

    # Format system prompt with current date and query classification
    formatted_system_prompt = SYSTEM_PROMPT_ENTITY_IDENTIFIER.format(
        current_date=current_date,
        query_type=query_type,
        trip_type=trip_type,
        trip_purpose=trip_purpose
    )

    # Create entity_identifier agent with formatted prompt
    entity_identifier = create_agent(
        model=get_openai_model("gpt-4o-mini"),
        response_format=ProviderStrategy(TravelInquiry),
        system_prompt=formatted_system_prompt,
        tools=[web_search,calculator],
        middleware=[
            SummarizationMiddleware(
                model=get_openai_model("gpt-4o-mini"),
                max_tokens_before_summary=4000,
                messages_to_keep=20,
            ),
            ToolCallLimitMiddleware(
                tool_name="web_search",
                run_limit=3,
            )
        ],
    )


    # Invoke entity_identifier with full message history
    inquiry_result = None
    try:
        result = await entity_identifier.ainvoke(
            {"messages": agent_messages},
            config=config,
        )

        if isinstance(result, dict) and "structured_response" in result:
            inquiry_result = result.get("structured_response")
        else:
            inquiry_result = result

        if inquiry_result is None:
            raise ValueError(
                f"Missing 'structured_response' in validator response."
            )

        if not isinstance(inquiry_result, TravelInquiry):
            raise ValueError(
                f"Invalid response type from validator agent. "
                f"Expected TravelInquiry, got {type(inquiry_result).__name__}"
            )
    except (APITimeoutError, httpx.ReadTimeout, httpx.TimeoutException) as e:
        logger.error(f"Timeout error in entity identifier: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but the request timed out while processing your query. "
            "This can happen when validation takes longer than expected.\n\n"
            "Please try again with your request, or you can:\n"
            "• Simplify your query\n"
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
                "reply_to_user": error_message,
            },
            goto="entity_identifier"
        )
    except (APIError, Exception) as e:
        logger.error(f"Error in entity identifier: {e}")
        # Return error message to user and allow retry
        error_message = (
            "⚠️ I'm sorry, but I encountered an error while processing your query. "
            "This might be a temporary issue.\n\n"
            "Please try again with your request, or you can:\n"
            "• Rephrase your request\n"
            "• Try a different query\n\n"
            "Would you like to try again?"
        )
        agent_messages.append(AIMessage(content=error_message))
        user_input = interrupt(error_message)
        agent_messages.append(HumanMessage(content=user_input))
        return Command(
            update={
                "messages": agent_messages,
                "user_feedback": user_input,
                "reply_to_user": error_message,
            },
            goto="entity_identifier"
        )

    # Handle informational queries (query_type="informational")
    # For informational queries, the agent generates an answer and ends the flow
    if query_type == "informational" and inquiry_result.reply_to_user:
        # Append AI message with the answer to conversation history
        agent_messages.append(AIMessage(content=inquiry_result.reply_to_user))
        
        # Return with answer and end the flow
        return Command(
            update={
                "reply_to_user": inquiry_result.reply_to_user,
                "current_state": "informational",
                "regenerated_question": inquiry_result.regenerated_question or "",
                "messages": agent_messages,
                "user_feedback": ""
            },
            goto=END
        )
    
    # Handle missing entities (reply_to_user is populated when asking for missing info)
    if inquiry_result.reply_to_user:
        # Append AI message to conversation history before interrupt
        agent_messages.append(AIMessage(content=inquiry_result.reply_to_user))

        # Interrupt and wait for user response
        user_input = interrupt(inquiry_result.reply_to_user)

        # When graph resumes with Command(resume=value), execution continues HERE
        # The resume value is passed to user_input variable
        # Append user's response to conversation history
        agent_messages.append(HumanMessage(content=user_input))

        # Loop back to entity_identifier to process the user's response
        return Command(
            update={
                "current_state": "entity_identifier",
                "reply_to_user": "",
                "regenerated_question": inquiry_result.regenerated_question or "",
                "messages": agent_messages,
                "user_feedback": user_input,
            },
            goto="entity_identifier"
        )

    # All entities collected - return with regenerated question
    # For trip_planning, proceed to transport; for specific_service, also proceed to transport
    return Command(
        update={
            "reply_to_user": "",
            "number_of_travelers": inquiry_result.number_of_travelers,
            "current_state": "transport",
            "regenerated_question": inquiry_result.regenerated_question,
            "messages": agent_messages,
            "user_feedback": ""
        }
    )

