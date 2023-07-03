# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import re
from typing import Any, Text, Dict, List, Union

import langchain
from langchain import LLMChain
from langchain.agents import initialize_agent, AgentType, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage, AgentAction, AgentFinish
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
        print('Inside Custom Action')
        print(selected_month)
        print(selected_place)
        print(selected_days)
        if not selected_month:
            return dispatcher.utter_message(text="In which month would you like to travel?")
        if not selected_place:
            return dispatcher.utter_message(text="where would you like to go?")
        if not selected_days:
            return dispatcher.utter_message(text="For how many days you would like to travel?")
        latest_message = f"I'm looking for recommendations and suggestions for things to do in destination {selected_place} during {selected_month} month for a {selected_days} days trip. Can you help me create an itinerary?"
        prompt = get_prompt(latest_message)
        message = get_itinerary(prompt)
        # print(message)
        dispatcher.utter_message(text=message)
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
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search


def get_itinerary(input: str):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    tools = [
        Tool(
            name="Search the internet",
            func=search_general,
            description="useful for when you need to search some information that is missing or you are not able to understand",
        )
    ]

    # Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
    template = """Answer the following questions as an expert travel planner.
    You have access to the following tools:
    
    {tools}
    
    Use the following format:
    Thought: Do I need to use a tool? Yes
    Question: the input question you must answer
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action

    Thought: Do I need to use a tool? No
    Final Answer: the final answer to the original input question
    
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
        stop=["\nObservation:", "\nFinal Answer:"],
        allowed_tools=tool_names
    )

    # Initiate the agent that will respond to our queries
    # Set verbose=True to share the CoT reasoning the LLM goes through
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)
    agent_response = agent_executor.run(input)
    print("Got the result from OpenAI")
    return agent_response


def get_prompt(user_input: str):
    prompt = f"""
    - You are an expert travel planner.
    - Your task is to extract relevant information from the UserInput below, delimited by triple backticks and help me prepare a travel itinerary.
    - Travel itinerary should be provided day wise.
    - Please consider the user preference such as month of travel, destination, number of days of travel, themes and any other preferences provided in the context.
    - The output should be in raw markdown for javascript format in Paragraph style.
    - Don't mention based on the user input in the response.
    - Don't mention Based on the UserInput in the response
    - Don't mention As an expert travel planner in the response
    - Don't specify Yes, I can help you create an itinerary in the response
    - Present the output in raw markdown for javascript format
    UserInput: ```{user_input}```
    """
    return prompt
