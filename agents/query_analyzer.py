from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agents.prompts import SYSTEM_PROMPT_QUERY_ANALYZER
from models.agent_models import TravelIntent, TravelAgentState
from tools.custom_tools import web_search
from utils import get_openai_model


async def analyze_query(state: TravelAgentState, config: RunnableConfig):
    """
    Analyzes the user's travel query to extract query type, trip type, and trip purpose.
    
    This function runs after the validator and analyzes the travel query to determine:
    - **Query Type**: trip_planning, informational, specific_service, or null
    - **Trip Type**: weekend_trip, within_country_trip, across_country_trip, or null
    - **Trip Purpose**: holy_trip, family_trip, adventure, relaxation, business_trip, cultural, 
                        honeymoon, food_tourism, bag_pack, or null
    
    The query analyzer uses the regenerated query from the validator (which may be a standalone 
    version of a follow-up question with full context) and invokes an LLM-based agent to 
    extract these characteristics. These classifications are then used by the entity_identifier 
    to determine which entities are required and how to validate them.
    
    Args:
        state: TravelAgentState containing:
            - regenerated_question: Travel query from validator (standalone with context if follow-up)
            - user_query: Fallback if regenerated_question is not available
        config: Runnable configuration for the agent
        
    Returns:
        Command object with state updates:
        - query_type: Classified query type (string or None)
        - trip_type: Classified trip type (string or None)
        - trip_purpose: Classified trip purpose (string or None)
        - Routing: Proceeds to entity_identifier (no goto, graph handles routing)
        
    Raises:
        ValueError: If no query is provided or if classifier response is invalid
    """
    print(f'Inside query-analyzer agent: {state}')
    # Get regenerated question (fully qualified query with all entities)
    # Fallback to user_query if regenerated_question is not available
    user_query = state.get("regenerated_question") or state.get("user_query", "")
    
    if not user_query:
        raise ValueError("No user query provided for intent classification")
    
    # Format system prompt with user_query only (entities not needed)
    formatted_prompt = SYSTEM_PROMPT_QUERY_ANALYZER.format(
        user_query=user_query
    )
    
    # Create query analyzer agent with formatted prompt
    query_analyzer_agent = create_agent(
        model=get_openai_model("gpt-4o-mini"),
        response_format=ProviderStrategy(TravelIntent),
        system_prompt=formatted_prompt,
        tools=[web_search],
    )
    
    # Invoke query analyzer agent
    result = await query_analyzer_agent.ainvoke(
        input={},
        config=config
    )
    
    if isinstance(result, dict) and "structured_response" in result:
        intent = result.get("structured_response")
    else:
        intent = result
    
    if intent is None:
        raise ValueError(
            f"Missing 'structured_response' in query analyzer response."
        )
    
    if not isinstance(intent, TravelIntent):
        raise ValueError(
            f"Invalid response type from query analyzer agent. "
            f"Expected TravelIntent, got {type(intent).__name__}"
        )
    
    # Extract query type, trip type and purpose values
    query_type = intent.query_type.value if intent.query_type else None
    trip_type = intent.trip_type.value if intent.trip_type else None
    trip_purpose = intent.trip_purpose.value if intent.trip_purpose else None
    
    # Return Command with state updates
    return Command(
        update={
            "query_type": query_type,
            "trip_type": trip_type,
            "trip_purpose": trip_purpose,
            "current_state": 'entity_identifier',
        },
    )