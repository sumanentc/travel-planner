# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import json
import re
from typing import Any, Text, Dict, List, Union
from rasa_sdk.events import SlotSet

import langchain
from langchain import LLMChain
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AgentAction, AgentFinish, SystemMessage
from langchain.tools import DuckDuckGoSearchRun, Tool
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class TravelItineraryAction(Action):
    def name(self) -> Text:
        return "travel_itinerary_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        selected_month = tracker.get_slot('month')
        selected_place = tracker.get_slot('place')
        selected_days = tracker.get_slot('days')
        inp_message = tracker.latest_message['text']
        print('Inside TravelItineraryAction')
        print(selected_month)
        print(selected_place)
        print(selected_days)
        if not selected_month:
            return dispatcher.utter_message(text="In which month would you like to travel?")
        if not selected_place:
            return dispatcher.utter_message(text="where would you like to go?")
        if not selected_days:
            return dispatcher.utter_message(text="For how many days you would like to travel?")
        latest_message = f"I'm looking for recommendations and suggestions for things to do in destination {selected_place} during {selected_month} month for a {selected_days} days trip. {inp_message} Can you help me create an itinerary?"
        prompt = get_itinerary_prompt(latest_message)
        message = get_response(prompt)
        dispatcher.utter_message(text=message)
        return []


class TravelQueryAction(Action):
    def name(self) -> Text:
        return "travel_query_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('Inside TravelQueryAction')
        inp_message = tracker.latest_message['text']
        selected_month = tracker.get_slot('month')
        selected_place = tracker.get_slot('place')
        travel_check_prompt = get_travel_query_check_prompt(inp_message)
        resp = run(travel_check_prompt)
        print(resp)
        if not resp['travel_query']:
            dispatcher.utter_message(response='utter_please_rephrase')
        else:
            if selected_place:
                inp_message = f'{inp_message}. Travel destination is {selected_place}.'
                prompt = get_travel_month_prompt(inp_message)
            else:
                if selected_month:
                    inp_message = f'{inp_message}. Month to travel {selected_month}.'
                prompt = get_travel_query_prompt(inp_message)
            message = get_response(prompt)
            dispatcher.utter_message(text=message)
        return []


class HotelQueryAction(Action):
    def name(self) -> Text:
        return "hotel_query_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('Inside HotelQueryAction')
        inp_message = tracker.latest_message['text']
        selected_month = tracker.get_slot('month')
        selected_place = tracker.get_slot('place')
        travel_check_prompt = get_hotel_query_check_prompt(inp_message)
        resp = run(travel_check_prompt)
        print(resp)
        if not resp['hotel_query']:
            dispatcher.utter_message(response='utter_please_rephrase')
        else:
            if selected_place and selected_month:
                inp_message = (
                    f'{inp_message}. Travel destination is {selected_place}.'
                    + f'. Month to travel {selected_month}.'
                )
            elif selected_place:
                inp_message = f'{inp_message}. Travel destination is {selected_place}.'
            elif selected_month:
                inp_message = f'{inp_message}. Month to travel {selected_month}.'
            prompt = get_hotel_enquiry_prompt(inp_message)
            message = get_response(prompt)
            dispatcher.utter_message(text=message)
        return []


class ActionUnlikelyIntent(Action):
    def name(self) -> Text:
        return "action_unlikely_intent"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        print(tracker.latest_message)
        print(tracker.slots)
        if current_intent := tracker.latest_message.get('intent'):
            current_intent_name = current_intent.get('name')
            print(current_intent_name)
            if current_intent_name == "nlu_fallback":
                user_input = tracker.latest_message.get('text')
                if requested_slot := tracker.slots.get('requested_slot'):
                    if requested_slot == 'month':
                        month_check_prompt = get_month_check_prompt(user_input)
                        resp = run(month_check_prompt)
                        if resp and not resp['valid_month']:
                            dispatcher.utter_message(response='utter_please_rephrase')
                    elif requested_slot == 'place':
                        place_check_prompt = get_place_check_prompt(user_input)
                        resp = run(place_check_prompt)
                        print(resp)
                        if resp and not resp['valid_place']:
                            dispatcher.utter_message(response='utter_please_rephrase')
                        else:
                            return [SlotSet("place", resp['place'])]
                    elif requested_slot == 'days':
                        days_check_prompt = get_day_check_prompt(user_input)
                        resp = run(days_check_prompt)
                        if resp and not resp['valid_days']:
                            dispatcher.utter_message(response='utter_please_rephrase')
                else:
                    travel_check_prompt = get_travel_query_check_prompt(user_input)
                    resp = run(travel_check_prompt)
                    if not resp['travel_query']:
                        dispatcher.utter_message(response='utter_please_rephrase')
        return []


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)

        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        # regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match[1].strip()
        action_input = match[2]

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_general(input_text):
    return DuckDuckGoSearchRun().run(f"{input_text}")


def search_online(input_text):
    return DuckDuckGoSearchRun().run(
        f"site:tripadvisor.com things to do{input_text}"
    )


def search_hotel(input_text):
    return DuckDuckGoSearchRun().run(
        f"site:booking.com hotels or resorts to stay {input_text}"
    )


def search_flight(input_text):
    return DuckDuckGoSearchRun().run(f"site:skyscanner.com {input_text}")


def get_response(input: str):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613')
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    tools = [
        Tool(
            name="Search the internet",
            func=search_general,
            description="useful for when you need to search some information that is missing or you are not able to understand",
        ),
        Tool(
            name="Search for hotels",
            func=search_hotel,
            description="useful for when you need to answer hotel related questions"
        ),
        Tool(
            name="Search tripadvisor",
            func=search_online,
            description="useful for when you need to answer trip plan questions"
        ),
    ]

    # https://github.com/hwchase17/langchain/issues/1358

    # Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
    template = """Answer the following questions as an expert travel planner to the best of your ability.
    You have access to the following tools:
    
    {tools}
    
    Use the following format:
    - Question: the input question you must answer
    - Thought: Do I need to use a tool? Yes
    - Action: the action to take, should be one of [{tool_names}]
    - Action Input: Provide the input required for the chosen tool, 
    - Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the following format(the prefix of "Thought: " and "Final Answer: " are must be included):
    
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]

    Begin! Remember to speak as a passionate and an expert travel planner when giving your final answer.
    
    Question: {input}
    {agent_scratchpad}"""

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Using tools, the LLM chain and output_parser to make an agent
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        # We use "Observation" as our stop sequence so it will stop when it receives Tool output
        # If you change your prompt template you'll need to adjust this as well
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    # Initiate the agent that will respond to our queries
    # Set verbose=True to share the CoT reasoning the LLM goes through
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,max_iterations=4, handle_parsing_errors="Check your output and make sure it conforms!")
    try:
        agent_response = agent_executor.run(input)
    except ValueError as e:
        print(e)
        response = str(e)
        agent_response = (
            response.removeprefix(
                "Could not parse LLM output: `"
            ).removesuffix("`")
            if response.startswith("Could not parse LLM output: `")
            else "Sorry, Couldn't help you at this moment. Please try sometime later"
        )
    except Exception as e:
        print(e)
        agent_response = "Sorry, Couldn't help you at this moment. Please try sometime later"
    print("Got the result from OpenAI")
    return agent_response


def get_itinerary_prompt(user_input: str):
    return f"""
    - You are an expert travel planner.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and help me prepare a travel itinerary.
    - Travel itinerary should be provided day wise.
    - Please consider the user preference such as month of travel, destination, number of days of travel, themes and any other preferences provided in the context.
    - The output should be in raw markdown for javascript format in Paragraph style.
    - Please check the best month of travel to the input destination. If it is different from input month then also provide recommendation for the best month to travle to input destination.
    - Don't mention based on the user input, in the response.
    - Don't mention Based on the UserInput, in the response
    - Don't mention As an expert travel planner, in the response
    - Don't mention Yes, I can help you create an itinerary, in the response
    - Don't mention Based on the user input, in the response
    - Don't mention Based on your input, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Based on your input, in the response
    - Present the output in raw markdown for javascript format
    UserInput: ```{user_input}```
    """


def get_travel_query_prompt(user_input: str):
    return f"""
    - You are an expert travel planner.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and help me prepare a travel destination recommendation.
    - Provide at max 2 to 3 recommendations related to travel query.
    - Please consider the user preferences such as month of travel, themes and any other preferences provided in the UserInput.
    - Please mention the best time to travel to these destinations.
    - Don't mention based on the user input, in the response.
    - Don't mention Based on the UserInput, in the response
    - Don't mention As an expert travel planner, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Yes, I can help you in the response
    - Don't mention Based on the user input, in the response
    - Don't mention Based on your input, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Based on your input, in the response
    - The output must be in raw markdown for javascript format in Paragraph style.
    UserInput: ```{user_input}```
    """


def get_travel_month_prompt(user_input: str):
    return f"""
    - You are an expert travel planner.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and help me prepare a travel time recommendation.
    - Please consider the user preferences such as destination to travel, themes or any other preferences provided in the UserInput.
    - Please mention the best time to travel to these destinations.
    - The output should be in raw markdown for javascript format in Paragraph style.
    - Don't mention based on the user input, in the response.
    - Don't mention Based on the UserInput, in the response
    - Don't mention As an expert travel planner, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Yes, I can help you in the response
    - Don't mention Based on the user input, in the response
    - Don't mention Based on your input, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Based on your input, in the response
    - Present the output in raw markdown for javascript format
    UserInput: ```{user_input}```
    """


def get_hotel_enquiry_prompt(user_input: str):
    return f"""
    - You are an expert travel planner for Hotel, Resort or Villa bookings.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and perform the following actions:
    1. Search for Hotels or Resorts for a given destination with the given preference from User.
    2. Add some recent reviews if available.
    2. Add the stay cost of per day in the response
    - Don't mention based on the user input, in the response.
    - Don't mention Based on the UserInput, in the response
    - Don't mention As an expert travel planner, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Yes, I can help you in the response
    - Don't mention Based on the user input, in the response
    - Don't mention Based on your input, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Based on your input, in the response
    - The output should be in raw markdown for javascript format in Paragraph style.
    UserInput: ```{user_input}```
    """


def get_month_check_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - is there a valid month in the input text ? (True or False)
                - Is the input query related to travel enquiry ? (True or False)
                
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "valid_month" and "travel_query" as the keys.
                If the information isn't present, use "None" \
                as the value.
                Make your response as short as possible.
                Format the valid_month and  travel_query value as a boolean.
                Input Text: <{user_input}>
                """


def get_place_check_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - is there a valid place on the world present in the input text ? (True or False)
                - extract valid place name from input text ? (string or None)
                
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "valid_place" and "place" as the keys.
                If the information isn't present, use "None" \
                as the value.
                Make your response as short as possible.
                Format the valid_place value as a boolean.
                Input Text: <{user_input}>
                """


def get_day_check_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - is there a valid number of days in the input text ? True or False)
                
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "valid_days" as the key.
                If the information isn't present, use "None" \
                as the value.
                Make your response as short as possible.
                Format the valid_days as a boolean.
                Input Text: <{user_input}>
                """


def get_travel_query_check_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - Is the input query related to travel ? (True or False)
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "travel_query" as the key.
                If the information isn't present, use "None" \
                as the value.
                Make your response as short as possible.
                Format the travel_query value as a boolean.
                Input Text: <{user_input}>
                """


def get_hotel_query_check_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - Is the input query related to hotel enquiry ? (True or False)
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "hotel_query" as the key.
                If the information isn't present, use "None" \
                as the value.
                Make your response as short as possible.
                Format the hotel_query value as a boolean.
                Input Text: <{user_input}>
                """


def run(prompt: str):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    messages = [
        SystemMessage(
            content="You are a helpful assistant."
        ),
        HumanMessage(
            content=prompt
        ),
    ]
    try:
        response = llm(messages)
    except Exception as e:
        print(e)
        return json.loads("")
    return json.loads(response.content)
