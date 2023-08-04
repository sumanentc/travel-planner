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
from langchain import LLMChain, LLMMathChain
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AgentAction, AgentFinish, SystemMessage
from langchain.tools import DuckDuckGoSearchRun, Tool, AIPluginTool
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class TravelItineraryAction(Action):
    def name(self) -> Text:
        return "travel_itinerary_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        selected_month = tracker.get_slot('month')
        selected_place = tracker.get_slot('destination')
        selected_days = tracker.get_slot('days')
        inp_message = tracker.latest_message['text']
        print('Inside TravelItineraryAction')
        print(selected_month)
        print(selected_place)
        print(selected_days)
        if not selected_month:
            return dispatcher.utter_message(response='utter_ask_month')
        if not selected_place:
            return dispatcher.utter_message(response='utter_ask_destination')
        if not selected_days:
            return dispatcher.utter_message(response='utter_ask_days')
        latest_message = f"I'm looking for recommendations and suggestions for things to do in destination {selected_place} during {selected_month} month for a {selected_days} days trip with hotel stay. {inp_message} Can you help me create an day wise itinerary with hotel stay?"
        prompt = get_itinerary_prompt(latest_message)
        message = get_llm_response(prompt)
        dispatcher.utter_message(text=message)
        return [SlotSet("preffered_itinerary", message)]


class TravelCostAction(Action):
    def name(self) -> Text:
        return "travel_cost_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        selected_month = tracker.get_slot('month')
        selected_source = tracker.get_slot('source')
        selected_destination = tracker.get_slot('destination')
        selected_days = tracker.get_slot('days')
        head_count = tracker.get_slot('head_count')
        selected_itinerary = tracker.get_slot('preffered_itinerary')
        inp_message = tracker.latest_message['text']
        print('Inside TravelCostAction')
        print(selected_month)
        print(selected_source)
        print(selected_destination)
        print(selected_days)
        print(head_count)
        if not selected_month:
            return dispatcher.utter_message(response='utter_ask_month')
        if not selected_destination:
            return dispatcher.utter_message(response='travel_query_action')
        if not selected_days:
            return dispatcher.utter_message(response='utter_ask_days')
        if not selected_source:
            return dispatcher.utter_message(response='utter_ask_source')
        if not head_count:
            return dispatcher.utter_message(response='utter_ask_head_count')

        if selected_itinerary:
            stay_location_prompt = get_hotel_names_prompt(selected_itinerary)
            resp = run(stay_location_prompt)
        if resp and resp['valid_stay']:
            preffered_stay = resp['stay']
            print(preffered_stay)

        latest_message = f"I'm looking for overall cost of my trip expenses from departure: {selected_source} to destination: {selected_destination} during the month: {selected_month} for {selected_days} days of a trip." \
                         f"Number of people travelling is {head_count}. Itinerary: {selected_itinerary}." \
                         f"Instruction from user {inp_message} "
        prompt = get_trip_cost_prompt(latest_message)
        message = get_llm_response(prompt)
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
        selected_place = tracker.get_slot('destination')
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
            message = get_llm_response(prompt)
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
        selected_place = tracker.get_slot('destination')
        selected_itinerary = tracker.get_slot('preffered_itinerary')
        travel_check_prompt = get_hotel_query_check_prompt(inp_message)
        resp = run(travel_check_prompt)
        # print(resp)
        if not resp['hotel_query']:
            dispatcher.utter_message(response='utter_please_rephrase')
        else:
            if selected_itinerary:
                inp_message = (
                    f'{inp_message}. Travel itinerary: {selected_itinerary}.'
                )
            elif selected_place and selected_month:
                inp_message = (
                        f'{inp_message}. Travel destination is {selected_place}.'
                        + f'. Month to travel {selected_month}.'
                )
            elif selected_place:
                inp_message = f'{inp_message}. Travel destination is {selected_place}.'
            elif selected_month:
                inp_message = f'{inp_message}. Month to travel {selected_month}.'
            prompt = get_hotel_enquiry_prompt(inp_message)
            message = get_llm_response(prompt)
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
                    elif requested_slot == 'destination':
                        place_check_prompt = get_place_check_prompt(user_input)
                        resp = run(place_check_prompt)
                        print(resp)
                        if resp and not resp['valid_place']:
                            dispatcher.utter_message(response='utter_please_rephrase')
                        else:
                            return [SlotSet("destination", resp['destination'])]
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
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        # regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
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
        f"site:tripadvisor.com {input_text}"
    )


def search_hotel(input_text):
    return DuckDuckGoSearchRun().run(
        f"site:booking.com prices of {input_text}"
    )


def search_flight(input_text):
    return DuckDuckGoSearchRun().run(f" Best Flight fares {input_text}")


def _handle_error(error) -> str:
    print(error)
    return str(error)[:50]


def get_llm_response(input: str):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613')
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
    tools = [
        Tool(
            name="Search the internet",
            func=search_general,
            description="useful for when you need to search some information that is missing or you are not able to understand",
        ),
        Tool(
            name="Search for hotels reviews and stay related queries",
            func=search_online,
            description="useful for when you need to search for hotels availability or reviews of some hotels"
        ),
        Tool(
            name="Search flight",
            func=search_flight,
            description="useful for when you need to answer flight questions"
        ),
        Tool(
            name="Search for Hotel or Resort cost or fares",
            func=search_hotel,
            description="useful for when you need to find the cost or fares of hotel stay"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to calculate something"
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
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=10,
                                                        handle_parsing_errors="Check your output and make sure it conforms!")
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
    - Search for hotels,resort or places to stay for each of the recommended places.
    - Must include recommended hotel or resort name for each day.
    - Must include the Hotel check-in and check-out activity for each and every hotel in the itinerary.
    - Please consider the user preference such as month of travel, destination, number of days of travel, themes and any other preferences provided in the context.
    - Please check the best month of travel to the input destination. If it is different from input month then also provide recommendation for the best month to travel to input destination.
    - Don't mention "As an expert travel planner", in the response.
    - Don't mention "based on the user input", in the response.
    - Don't mention "Based on the UserInput", in the response
    - Don't mention "As an expert travel planner", in the response
    - Don't mention "Yes, I can help you create an itinerary", in the response
    - Don't mention "Based on your input", in the response
    - Don't mention "Based on the user's preference", in the response
    - Don't mention "Based on your preferences", in the response
    - Don't mention "Based on the search results", in the response
    - The output must be in raw markdown for javascript format in Paragraph style.
    UserInput: ```{user_input}```
    """


def get_travel_query_prompt(user_input: str):
    return f"""
    - You are an expert travel planner.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and help me prepare a travel destination recommendation.
    - Provide at max 2 to 3 recommendations related to travel query.
    - Please consider the user preferences such as month of travel, themes and any other preferences provided in the UserInput.
    - Please mention the best time to travel to these destinations.
    - Don't mention "Based on the user's preference", in the response.
    - Don't mention "Based on your query", in the response.
    - Don't mention "As an expert travel planner", in the response.
    - Don't mention "based on the user input", in the response.
    - Don't mention "Based on the UserInput", in the response
    - Don't mention "As an expert travel planner", in the response
    - Don't mention "Based on the user's preference", in the response
    - Don't mention "Yes, I can help you" in the response
    - Don't mention "Based on your input", in the response
    - Don't mention "Based on the user's" preference, in the response
    - Don't mention "Based on your input", in the response
    - The output must be in raw markdown for javascript format in Paragraph style.
    UserInput: ```{user_input}```
    """


def get_travel_month_prompt(user_input: str):
    return f"""
    - You are an expert travel planner.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and help me prepare a travel time recommendation.
    - Please consider the user preferences such as destination to travel, themes or any other preferences provided in the UserInput.
    - Please mention the best time to travel to these destinations.
    - Don't mention "Based on the user's preference", in the response.
    - Don't mention As an expert travel planner, in the response.
    - Don't mention based on the user input, in the response.
    - Don't mention Based on the UserInput, in the response
    - Don't mention As an expert travel planner, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Yes, I can help you in the response
    - Don't mention Based on the user input, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Based on your input, in the response
    - The output should be in raw markdown for javascript format in Paragraph style.
    UserInput: ```{user_input}```
    """


def get_hotel_enquiry_prompt(user_input: str):
    return f"""
    - You are an expert travel planner for Hotel, Resort or Villa bookings.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and help me provide hotel recommendations:
    - Suggest Hotels or Resorts according to the given itinerary or destination with the given preference if provided like budget in the input.
    - If month is provided make sure the recommended Hotels or Resorts are available in that month.
    - Add some recent reviews of each hotels if available.
    - Don't mention As an expert travel planner, in the response.
    - Don't mention based on the user input, in the response.
    - Don't mention Based on the UserInput, in the response
    - Don't mention As an expert travel planner, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Yes, I can help you in the response
    - Don't mention Based on your input, in the response
    - Don't mention Based on the user's preference, in the response
    - Don't mention Based on your input, in the response
    - The output should be in raw markdown for javascript format in Paragraph style.
    UserInput: ```{user_input}```
    """


def get_trip_cost_prompt(user_input: str):
    return f"""
    - You are an expert travel planner who helps with the trip cost calculation and estimation.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and perform the following actions to calulate the trip cost:
    - Search for flight from departure to destination in the input month.
    - Calculate the total flight fares by considering the above flight fares and number of people present in the trip.
    - Extract the hotel names and number of days stay in the itinerary.
    - Calculate the hotel cost for all of the above selected hotels .
    - calculate the total hotel cost considering cost of all the hotels.
    - Calculate the overall trip cost with breakup which include flight cost and hotel cost for each and every hotel.
    - Don't mention "To calculate the hotel cost" or "To calculate the overall trip cost", in the response.
    - Don't mention "To calculate the overall trip cost", in the response.
    - Don't mention "As an expert travel planner", in the response.
    - Don't mention "based on the user input", in the response.
    - Don't mention "Based on the UserInput", in the response
    - Don't mention "Based on the user's preference", in the response
    - Don't mention "Yes, I can help you" in the response
    - Don't mention "Based on your input", in the response
    - Don't mention "Based on the user's preference", in the response
    - Don't mention "Based on your input", in the response
    - Don't mention "Based on the information provided", in the response
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


def get_flight_cost_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - extract valid departure and destination place from the input text ? (string or None)
                - extract valid head count from the input text ? (string or None)
                - Search for flight from departure to destination in the input month.
                - Calculate the flight cost by considering the above flight fares * 2 * head count extracted above.
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "flight_cost" and "head_count" as the keys.
                If the information isn't present, use "None" \
                as the value.
                Make your response as short as possible.
                Format the flight_cost and  head_count value as a string.
                Input Text: <{user_input}>
                """


def get_place_check_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - is there a valid place on the world present in the input text ? (True or False)
                - extract valid place name from input text ? (string or None)
                
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "valid_place" and "destination" as the keys.
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


def get_hotel_names_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - is there a valid hotel names present in the input text ? (True or False)
                - Extract valid hotel names for each and every day from input text
                - return stay as hotel name day wise in the response.(string or None)
                
                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "valid_stay" and "stay" as the keys.
                If the information isn't present, use "None" \
                as the value.
                Make your response as short as possible.
                Format the valid_stay value as a boolean.
                Format the stay value as a list of string.
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
