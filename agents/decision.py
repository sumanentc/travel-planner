from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from agents.prompts import CONFIRMATION_PROMPT
from models.agent_models import TravelAgentState, NextStep
from utils import get_openai_model


async def decide_next_step(state: TravelAgentState, config: RunnableConfig):
    """
    Decision node that asks user for confirmation and checks their reply.
    
    This function is called after each major agent (transport, accommodation, itinerary)
    completes its work. It follows a 2-step process:
    1. First call: Send confirmation message to user based on current_state
    2. Second call (after user reply): Use LLM to check if user wants to proceed
    
    Args:
        state (TravelAgentState): Current state containing:
            - current_state: The current stage (transport, accommodation, itinerary)
            - messages: Conversation history
            - user_feedback: Latest user response
        config (RunnableConfig): Runtime configuration
    
    Returns:
        Command: LangGraph command with updated state
        
    Workflow:
        1. Determine which stage we're at (transport/accommodation/itinerary)
        2. Check if we already asked for confirmation (by looking at last AI message)
        3. If not asked yet: Send confirmation question and wait for user reply
        4. If already asked: Use LLM to analyze user's reply
        5. If confirmed: route to next agent, else end flow
    """
    print(f'Inside decide_next_step: current_state={state.get("current_state")}')
    
    current_state = state.get("current_state", "")
    messages = state.get("messages", [])
    user_feedback = state.get("user_feedback", "")
    
    # Determine which stage we're at from current_state set by the previous agent
    stage = None
    if "transport" in current_state:
        stage = "transport"
    elif "accommodation" in current_state:
        stage = "accommodation"
    elif "itinerary" in current_state or "planning" in current_state:
        stage = "itinerary"
    elif "email" in current_state:
        stage = "email"
    
    if not stage:
        print(f"Unknown stage for current_state: {current_state}")
        return Command(
            update={
                "reply_to_user": "Unable to determine next step. Please try again."
            }
        )
    
    # Define confirmation questions for each stage
    confirmation_questions = {
        "transport": "Great! Now that we've sorted out your transportation, shall we look for accommodation options for your stay?",
        "accommodation": "Perfect! Your accommodation is all set. Would you like me to create a detailed day-by-day itinerary for your trip?",
        "itinerary": "Wonderful! Your complete itinerary is ready. Would you like me to email it to you so you have it handy during your trip?",
        "email": "Would you like any more help with your trip planning?"
    }
    
    agent_messages = messages.copy() if messages else []
    
    # Check if we have user_feedback (meaning user already responded)
    if not user_feedback:
        # Step 1: Ask user for confirmation
        print(f'Asking user for confirmation at {stage} stage')
        ask_confirmation = confirmation_questions.get(stage, "Would you like to proceed to the next step?")
        agent_messages.append(AIMessage(content=ask_confirmation))
        
        user_input = interrupt(ask_confirmation)
        agent_messages.append(HumanMessage(content=user_input))
        
        # Set user_feedback so we process it in the same execution
        user_feedback = user_input
    
    # Step 2: User has replied (user_feedback exists), use LLM to check their response
    print(f'Analyzing user reply for {stage} stage')
    
    # Use the generic confirmation prompt to validate user feedback
    system_prompt = CONFIRMATION_PROMPT
    
    # Create decision agent
    decision_agent = create_agent(
        model=get_openai_model("gpt-4o-mini"),
        tools=[],
        system_prompt=system_prompt,
        response_format=ProviderStrategy(NextStep),
    )
    
    # Invoke decision agent to check if user confirmed
    try:
        result = await decision_agent.ainvoke(
            {"messages": agent_messages},
            config=config
        )
        
        # Extract structured response
        if isinstance(result, dict) and "structured_response" in result:
            decision = result.get("structured_response")
        else:
            decision = result
        
        if not isinstance(decision, NextStep):
            raise ValueError(f"Expected NextStep, got {type(decision)}")
        
        print(f'Decision: user_confirmation={decision.user_confirmation}, reasoning={decision.reasoning}')
        
    except Exception as e:
        print(f"Error in decision agent: {e}")
        return Command(
            update={
                "reply_to_user": "Unable to process your response. Please try again.",
                "messages": agent_messages
            }
        )
    
    # If user confirmed, set current_state to indicate next agent
    # Graph will use conditional edges to route based on current_state
    if decision.user_confirmation:
        # Map stage to next state value
        next_state_map = {
            "transport": "accommodation",
            "accommodation": "itinerary",
            "itinerary": "email",
            "email": "validator"  # After email, if user wants help, go to validator
        }
        next_state = next_state_map.get(stage)
        
        if not next_state:
            print(f'No next state found for stage: {stage}')
            return Command(
                update={
                    "reply_to_user": "Unable to determine next step.",
                    "messages": agent_messages,
                    "current_state": "",
                    "user_feedback": ""
                }
            )
        
        # Special handling for email stage - route directly to intent_validator
        if stage == "email":
            print(f'User wants more help after email, routing to intent_validator')
            return Command(
                update={
                    "messages": agent_messages,
                    "current_state": "",
                    "user_feedback": user_feedback
                },
                goto="intent_validator"
            )
        
        print(f'User confirmed, setting current_state to {next_state}')
        return Command(
            update={
                "messages": agent_messages,
                "current_state": next_state,
                "user_feedback": ""
            }
        )
    
    # If user said no, end the flow with a farewell message
    print(f'User declined to proceed, ending flow')
    farewell_msg = "Thank you for using RoamWise! Have a wonderful trip! 🌍✈️" if stage == "email" else "Thank you for using our travel planning service. Feel free to reach out if you need any assistance in the future!"
    agent_messages.append(AIMessage(content=farewell_msg))
    
    from langgraph.constants import END
    return Command(
        update={
            "reply_to_user": farewell_msg,
            "messages": agent_messages,
            "current_state": "",
            "user_feedback": ""
        },
        goto=END
    )
