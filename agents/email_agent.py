from __future__ import print_function

import os
import traceback

import sib_api_v3_sdk
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.types import Command, interrupt
from sib_api_v3_sdk.rest import ApiException

from agents.prompts import SYSTEM_PROMPT_EMAIL_AGENT
from models.agent_models import TravelAgentState, EmailContent
from utils import get_openai_model


async def send_email(state: TravelAgentState, config: RunnableConfig):
    """
    Email agent that handles sending travel itineraries via email.
    
    Workflow:
    1. Check if email_id is present in state
    2. If not, check conversation history for email pattern
    3. If still not found, ask user for email address
    4. Validate email pattern
    5. Generate HTML email content and send
    
    Args:
        state (TravelAgentState): Current state containing:
            - responses: Dict with itinerary_details
            - email_id: User's email address (if provided)
            - messages: Conversation history
            - trip_type: Type of trip
            - trip_purpose: Purpose of trip
            - user_feedback: User's response to previous question
        config (RunnableConfig): Runtime configuration for the agent
    
    Returns:
        Command: LangGraph command with updated state
    
    Raises:
        ValueError: If no itinerary found in responses
        ApiException: If email sending fails
    """
    print(f'Inside email-agent: {state}')
    
    import re
    
    # Extract data from state
    email_id = state.get("email_id")
    responses = state.get("responses", {})
    messages = state.get("messages", [])
    trip_type = state.get("trip_type", "unknown")
    trip_purpose = state.get("trip_purpose", "unknown")
    
    # Extract itinerary details from responses dict
    itinerary_details = responses.get("itinerary_details", "")
    
    # Check if itinerary exists
    if not itinerary_details:
        print('No itinerary found in responses')
        return Command(
            update={
                "reply_to_user": "No itinerary found to send. Please create an itinerary first.",
                "messages": messages,
                "user_feedback": ""
            },
            goto="itinerary_agent"
        )
    
    agent_messages = messages.copy() if messages else []
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Step 1: Check if email_id is present in state
    # If not, ask user directly for email address
    if not email_id:
        print('Asking user for email address')
        ask_email = "Please provide your email address so I can send you the itinerary."
        agent_messages.append(AIMessage(content=ask_email))
        
        user_input = interrupt(ask_email)
        agent_messages.append(HumanMessage(content=user_input))
        
        # Step 4: Validate email pattern
        found_emails = re.findall(email_pattern, user_input)
        
        if found_emails:
            email_id = found_emails[0]
            print(f'Extracted and validated email from user input: {email_id}')
        else:
            # Invalid email format - ask again
            print('Invalid email format provided')
            error_message = "❌ Invalid email format. Please provide a valid email address (e.g., user@example.com)."
            agent_messages.append(AIMessage(content=error_message))
            
            # Loop back to email_agent to ask again
            return Command(
                update={
                    "messages": agent_messages,
                    "user_feedback": error_message,
                },
                goto="email_agent"
            )
        
        # Loop back to email_agent with validated email_id
        return Command(
            update={
                "messages": agent_messages,
                "email_id": email_id,
                "user_feedback": user_input,
            },
            goto="email_agent"
        )
    
    # Step 3: Generate HTML email content and send
    print(f'Generating HTML email content for {email_id}')
    
    # Combine all details for email content
    transport_details = responses.get("transport_details", "")
    accommodation_details = responses.get("accommodation_details", "")
    
    full_itinerary_content = f"""
{transport_details}

{accommodation_details}

{itinerary_details}
""".strip()
    
    # Format system prompt with itinerary content
    formatted_system_prompt = SYSTEM_PROMPT_EMAIL_AGENT.format(
        itinerary_content=full_itinerary_content,
        trip_type=trip_type,
        trip_purpose=trip_purpose
    )
    
    # Create email agent to generate HTML content
    email_agent = create_agent(
        model=get_openai_model("gpt-4o-mini"),
        tools=[],
        system_prompt=formatted_system_prompt,
        response_format=ProviderStrategy(EmailContent),
    )
    
    # Invoke agent to generate HTML and subject
    email_result = None
    try:
        # Create a simple message asking to generate email
        generation_messages = [
            HumanMessage(content="Generate the HTML email content and subject line for this itinerary. Set email_sent to true.")
        ]
        
        result = await email_agent.ainvoke(
            {"messages": generation_messages},
            config=config
        )
        
        # Extract structured response
        if isinstance(result, dict) and "structured_response" in result:
            email_result = result.get("structured_response")
        else:
            email_result = result
        
        if not isinstance(email_result, EmailContent):
            raise ValueError(f"Expected EmailContent, got {type(email_result)}")
        
        print(f'Generated email - subject: {email_result.subject[:50]}...')
        
    except Exception as e:
        print(f"Error generating email content: {e}")
        traceback.print_exc()
        error_msg = "❌ Failed to generate email content. Please try again later or contact support."
        agent_messages.append(AIMessage(content=error_msg))
        return Command(
            update={
                "reply_to_user": "",
                "messages": agent_messages,
                "user_feedback": "",
                "current_state": ""
            },
            goto=END
        )
    
    # Validate generated content
    if not email_result.html_content or not email_result.subject:
        print('Failed to generate complete email content')
        error_msg = "❌ Failed to generate complete email content. Please try again later or contact support."
        agent_messages.append(AIMessage(content=error_msg))
        return Command(
            update={
                "reply_to_user": "",
                "messages": agent_messages,
                "user_feedback": "",
                "current_state": ""
            },
            goto=END
        )
    
    # Send the email
    print(f'Sending email to: {email_id}')
    
    # Configure API key authorization
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv('BREVO_API_KEY')
    
    # Validate API key
    if not configuration.api_key['api-key']:
        error_msg = "❌ Email service is not configured. Please contact support."
        agent_messages.append(AIMessage(content=error_msg))
        return Command(
            update={
                "reply_to_user": "",
                "messages": agent_messages,
                "user_feedback": "",
                "current_state": ""
            },
            goto=END
        )
    
    # Create an instance of the API class
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(configuration)
    )
    
    # Create email content with proper sender
    sender_email = os.getenv('SENDER_EMAIL', 'noreply@roamwise.com')
    sender_name = os.getenv('SENDER_NAME', 'RoamWise')
    
    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": email_id}],
        sender={"email": sender_email, "name": sender_name},
        subject=email_result.subject,
        html_content=email_result.html_content
    )
    
    # Send the email
    try:
        api_response = api_instance.send_transac_email(send_smtp_email)
        print(f'Email sent successfully to {email_id}')
        print(f'Brevo API Response: {api_response}')
        
        # Confirm email sent successfully
        success_message = f"✅ Great! I've sent your complete travel itinerary to {email_id}. Please check your inbox (and spam folder if you don't see it). Have a wonderful trip! 🎉"
        agent_messages.append(AIMessage(content=success_message))
        
        # Set current_state to 'email' so decision node can handle post-email flow
        return Command(
            update={
                "reply_to_user": "",
                "messages": agent_messages,
                "user_feedback": "",
                "current_state": "email"
            }
        )
            
    except ApiException as e:
        tb_str = traceback.format_exc()
        print("Exception caught while sending email!")
        print(tb_str)
        
        # Check if it's an authentication error
        if e.status == 401:
            error_msg = f"❌ Email service authentication failed. The API key is not enabled or invalid. Please contact support to configure the email service."
        else:
            error_msg = f"❌ Failed to send email to {email_id}. Error: {e.reason}. Please try again later or contact support."
        
        agent_messages.append(AIMessage(content=error_msg))
        
        return Command(
            update={
                "reply_to_user": "",
                "messages": agent_messages,
                "user_feedback": "",
                "current_state": ""
            },
            goto=END
        )