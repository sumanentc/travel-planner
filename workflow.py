from langgraph.constants import START, END
from langgraph.graph import StateGraph

from agents.accomodation import get_accomodation_details
from agents.query_analyzer import analyze_query
from agents.decision import decide_next_step
from agents.email_agent import send_email
from agents.itinerary import itinerary_planner
from agents.transport import get_transport_details
from agents.entity_identifier import  collect_missing_entities
from agents.validator import validate_intent
from models.agent_models import TravelAgentState

async def should_proceed_to_query_analyzer(state: TravelAgentState):
    """
    Determines if the inquiry should proceed to query analyzer.
    Returns True only when there is a regenerated question and no reply is pending for the user.
    """
    regenerated_question = state.get("regenerated_question", "")
    reply_to_user = state.get("reply_to_user", "")

    return True if regenerated_question and not reply_to_user else False


async def should_proceed_to_entity_identifier(state: TravelAgentState):
    """
    Determines if the inquiry should proceed to entity identifier agent.
    
    The entity identifier agent collects missing travel entities (destination, dates, 
    travelers, budget, preferences) required for trip planning.
    
    Args:
        state (TravelAgentState): Current state containing current_state and reply_to_user
    
    Returns:
        bool: True if should route to entity_identifier, False to end
        
    Routing Logic:
        - True: current_state contains 'entity_identifier' AND reply_to_user is empty
        - False: Otherwise (ends the flow or requires user interaction)
    """
    current_state = state.get("current_state", "")
    reply_to_user = state.get("reply_to_user", "")
    
    # Proceed to entity identifier only if classifier set current_state to 'entity_identifier' 
    # AND no reply is pending for the user
    return True if 'entity_identifier' in current_state and not reply_to_user else False

async def should_proceed_to_transport(state: TravelAgentState):
    """
    Determines if the inquiry should proceed to transport planning.
    Only proceed when all required entities are collected and no reply is pending.
    
    For informational queries (current_state="informational"), this returns False,
    routing to END since the answer is already in reply_to_user.
    """
    current_state = state.get("current_state", "")
    reply_to_user = state.get("reply_to_user", "")

    # Proceed to transport only if current_state is transport AND no reply is needed
    # For informational queries (current_state="informational"), returns False -> routes to END
    return 'transport' in current_state and not reply_to_user if current_state else False


async def route_from_transport(state: TravelAgentState):
    """
    Routes from transport agent based on whether user feedback is needed.
    If user_feedback exists, it means transport agent needs more input, so loop back.
    Otherwise, proceed to decision node.
    """
    user_feedback = state.get("user_feedback", "")
    
    # If user_feedback exists, transport agent is waiting for more input
    if user_feedback:
        return "transport_agent"
    else:
        return "decide_next_step"

async def route_from_accommodation(state: TravelAgentState):
    user_feedback = state.get("user_feedback", "")

    # If user_feedback exists, accommodation agent is waiting for more input
    if user_feedback:
        return "accommodation_agent"
    else:
        return "decide_next_step"

async def route_from_itinerary(state: TravelAgentState):
    user_feedback = state.get("user_feedback", "")

    # If user_feedback exists, itinerary agent is waiting for more input
    if user_feedback:
        return "itinerary_agent"
    else:
        return "decide_next_step"

async def route_from_email(state: TravelAgentState):
    user_feedback = state.get("user_feedback", "")

    # If user_feedback exists, email agent is waiting for more input (email validation)
    if user_feedback:
        return "email_agent"
    else:
        # Email sent successfully, route to decision node to check if user needs more help
        return "decide_next_step"

async def route_from_decision(state: TravelAgentState):
    """
    Routes from decision node based on current_state and user_feedback.
    
    This function checks if decision node needs to loop back to itself (when user_feedback exists)
    or route to the next agent (when current_state is set after user confirmation).
    
    Args:
        state (TravelAgentState): Current state with current_state and user_feedback fields
    
    Returns:
        str: Name of the next node to route to
    """
    current_state = state.get("current_state", "")
    reply_to_user = state.get("reply_to_user", "")
    user_feedback = state.get("user_feedback", "")
    
    # If user_feedback exists and current_state indicates we're still in confirmation stage,
    # loop back to decision node to process the feedback
    if user_feedback and current_state in ["transport", "accommodation", "itinerary", "email"]:
        return "decide_next_step"
    
    # If there's a reply to user, end the flow (user declined or error)
    if reply_to_user:
        return END
    
    # Map current_state to agent names
    if "accommodation" in current_state:
        return "accommodation_agent"
    elif "itinerary" in current_state or "planning" in current_state:
        return "itinerary_agent"
    elif "email" in current_state:
        return "email_agent"
    else:
        return END


def get_travel_agent(checkpointer, store):
    builder = StateGraph(TravelAgentState)
    builder.add_node("intent_validator", validate_intent)
    builder.add_node("query_analyzer", analyze_query)
    builder.add_node("entity_identifier", collect_missing_entities)
    builder.add_node("transport_agent", get_transport_details)
    builder.add_node("accommodation_agent", get_accomodation_details)
    builder.add_node("itinerary_agent", itinerary_planner)
    builder.add_node("email_agent", send_email)
    builder.add_node("decide_next_step", decide_next_step)  # Decision node
    
    # Start flow
    builder.add_edge(START, "intent_validator")
    
    # Intent validator -> Query Analyzer
    builder.add_conditional_edges(
        "intent_validator",
        should_proceed_to_query_analyzer,
        {True: "query_analyzer", False: END}
    )
    
    # Query Analyzer -> Entity Identifier
    builder.add_conditional_edges(
        "query_analyzer",
        should_proceed_to_entity_identifier,
        {True: "entity_identifier", False: END}
    )
    
    # Entity Identifier -> Transport
    builder.add_conditional_edges(
        "entity_identifier",
        should_proceed_to_transport,
        {True: "transport_agent", False: END}
    )
    
    # Transport -> Decision Node or loop back to itself
    builder.add_conditional_edges(
        "transport_agent",
        route_from_transport,
        {
            "transport_agent": "transport_agent",
            "decide_next_step": "decide_next_step"
        }
    )

    # Accommodation -> Conditional routing (loop back if user_feedback exists)
    builder.add_conditional_edges(
        "accommodation_agent",
        route_from_accommodation,
        {
            "accommodation_agent": "accommodation_agent",
            "decide_next_step": "decide_next_step"
        }
    )
    
    # Itinerary -> Conditional routing (loop back if user_feedback exists)
    builder.add_conditional_edges(
        "itinerary_agent",
        route_from_itinerary,
        {
            "itinerary_agent": "itinerary_agent",
            "decide_next_step": "decide_next_step"
        }
    )

    # Email -> Conditional routing (loop back if user_feedback exists)
    builder.add_conditional_edges(
        "email_agent",
        route_from_email,
        {
            "email_agent": "email_agent",
            "decide_next_step": "decide_next_step",
            END: END
        }
    )
    
    # Decision Node -> Routes to next agent based on current_state
    builder.add_conditional_edges(
        "decide_next_step",
        route_from_decision,
        {
            "decide_next_step": "decide_next_step",
            "accommodation_agent": "accommodation_agent",
            "itinerary_agent": "itinerary_agent",
            "email_agent": "email_agent",
            "intent_validator": "intent_validator",
            END: END
        }
    )

    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph
