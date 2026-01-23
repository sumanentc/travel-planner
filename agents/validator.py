import logging
from datetime import datetime

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from openai import APITimeoutError, APIError

from agents.prompts import SYSTEM_PROMPT_VALIDATOR
from models.agent_models import TravelAgentState, UserIntent
from tools.custom_tools import web_search
from utils import get_openai_model

logger = logging.getLogger(__name__)


async def validate_intent(state: TravelAgentState, config: RunnableConfig):
    """
    Validates user query intent and routes to appropriate next step based on query classification.
    
    This function acts as the entry point for all user queries, determining whether the query is:
    - Travel-related: Forwards to query_analyzer for further processing
    - Non-travel (chit-chat/out-of-context): Responds directly to user and ends conversation
    
    The validator also handles follow-up questions by regenerating them as standalone queries
    with full context from conversation history.
    
    Query Classification:
    - **travel**: Travel planning or informational queries → proceeds to query_analyzer
    - **chit-chat**: Casual conversation, greetings → responds and ends
    - **out-of-context**: Non-travel queries → responds and ends
    
    Args:
        state: TravelAgentState containing:
            - user_query: Initial user query (first interaction)
            - user_feedback: User response to previous question (if resuming from interrupt)
            - messages: Full conversation history
        config: Runnable configuration for the agent
        
    Returns:
        Command object with state updates:
        - For non-travel queries:
            * reply_to_user: Response message for user
            * regenerated_question: Empty or regenerated follow-up
            * is_followup: Boolean indicating if query references previous context
            * messages: Updated conversation history
            * Routing: Ends conversation (no goto)
        - For travel queries:
            * reply_to_user: Empty string
            * regenerated_question: Standalone travel query with full context
            * is_followup: Boolean indicating if query references previous context
            * messages: Updated conversation history
            * Routing: Proceeds to query_analyzer (no goto, graph handles routing)
    
    Error Handling:
        - Timeout errors: Prompts user to retry with simpler query
        - API errors: Prompts user to rephrase or try again
        - Both use interrupt() to get user input and loop back to validator
    """
    print(f'Inside validate intent: {state}')
    # Get user query, messages, and existing entities
    user_query = state.get("user_feedback", "") or state.get("user_query", "")
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

    # Get current date for prompt formatting
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Format system prompt with current date
    formatted_system_prompt = SYSTEM_PROMPT_VALIDATOR.format(current_date=current_date)

    # Create data_collector agent with formatted prompt
    intent_validator = create_agent(
        model=get_openai_model("gpt-4o-mini"),
        response_format=ProviderStrategy(UserIntent),
        system_prompt=formatted_system_prompt,
        tools=[web_search],
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

    # Invoke intent_validator with full message history
    try:
        response = await intent_validator.ainvoke(
            {"messages": agent_messages},
            config=config
        )
        
        inquiry_result = response.get("structured_response")
        
        if inquiry_result is None:
            raise ValueError(
                f"Missing 'structured_response' in validator response."
            )

        if not isinstance(inquiry_result, UserIntent):
            raise ValueError(
                f"Invalid response type from validator agent. "
                f"Expected UserIntent, got {type(inquiry_result).__name__}"
            )
    except (APITimeoutError, httpx.ReadTimeout, httpx.TimeoutException) as e:
        logger.error(f"Timeout error in intent validator: {e}")
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
            },
        )
    except (APIError, Exception) as e:
        logger.error(f"Error in intent validator: {e}")
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
            },
        )

    # Handle different inquiry types based on validator response
    
    # Case 1: user_input - asking for missing information, loop back to validator
    if inquiry_result.inquiry_type != "travel":
        # Append AI message to conversation history
        agent_messages.append(AIMessage(content=inquiry_result.reply_to_user))
        
        # Return with reply, ending the conversation
        return Command(
            update={
                "reply_to_user": inquiry_result.reply_to_user,
                "is_followup": inquiry_result.is_followup,
                "regenerated_question": inquiry_result.regenerated_question or "",
                "messages": agent_messages,
                "user_feedback": ""
            },
        )
    # Case 3: travel - pass regenerated question to data_collector
    else:  # inquiry_type == "travel"
        return Command(
            update={
                "reply_to_user": "",
                "is_followup": inquiry_result.is_followup,
                "regenerated_question": inquiry_result.regenerated_question,
                "messages": agent_messages,
                "user_feedback": ""
            }
        )