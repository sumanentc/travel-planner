import traceback
from typing import Annotated, Literal

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from gpt_researcher import GPTResearcher
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

Operator = Literal["+", "-", "*", "/"]

@tool("web_search", description="A tool for searching information on the web.")
def web_search(query: str,runtime: ToolRuntime) -> str:
    """Search the web for information."""
    try:
        # Use DDGS directly with text search for better reliability
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3,timeout=30,safesearch='off', timelimit='y',region='us-en',backend="duckduckgo"))
            if not results:
                return "No results found."
            
            # Format results as a list of dictionaries
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
            return str(formatted_results)
    except DDGSException as e:
        print(f"DuckDuckGo search error: {str(e)}")
        return "No results found. Please try a different search query."
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return f"Search failed: {str(e)}"

@tool("calculator", description="A tool for performing basic arithmetic operations (+, -, *, /). Returns error message for division by zero.")
def calculator(a: float, b: float, operator: Annotated[Operator, "operator"]) -> float | str:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        result = a * b
        if a == 0 or b == 0:
            return f"Warning: Multiplication by zero detected. Result: {result}"
        return result
    elif operator == "/":
        if b == 0:
            return "Error: Division by zero is undefined"
        return a / b
    else:
        return "Invalid operator. Please use one of: +, -, *, /"


@tool("transport_search", description="A tool for searching various transport options including flights, trains, buses, and cabs.")
async def transport_search(query: str) -> str:
    """
    Search the web for various transport options based on the given query.
    Returns a detailed research report with transport options, pricing, and booking information.
    
    Args:
        query: Search query for transport options (e.g., "flights from Mumbai to Goa on Dec 25")
    
    Returns:
        str: Research report with transport options and sources
    """
    try:
        # Initialize GPTResearcher with query and report type
        import os
        
        # Set environment variables for GPTResearcher configuration
        os.environ['RETRIEVER'] = 'duckduckgo'
        os.environ['SCRAPER'] = 'zendriver'
        
        researcher = GPTResearcher(
            query=query, 
            report_type='research_report',
            config_path=None,
            role="Travel agent helping to find transportation options"
        )
        
        print(f"Transport search - Using retriever: duckduckgo")
        print(f"Transport search - Using scraper: zendriver")
        
        # Conduct research
        await researcher.conduct_research()
        
        # Generate report
        report = await researcher.write_report()
        
        # Get additional context (optional - can be used for enhanced responses)
        research_sources = researcher.get_research_sources()
        research_images = researcher.get_research_images()
        
        print(f"Transport search - Found {len(research_sources)} sources, {len(research_images)} images")
        
        # Format response with report, sources, and images
        formatted_response = f"{report}\n\n"
        
        # Add sources if available
        if research_sources:
            formatted_response += f"Sources: {', '.join(research_sources[:5])}\n\n"
        
        # Add images if available
        if research_images:
            formatted_response += f"Images: {', '.join(research_images[:3])}"
        
        return formatted_response
        
    except Exception as ex:
        tb_str = traceback.format_exc()
        print(f"Transport search error: {ex}")
        print(tb_str)
        # Return error message instead of raising to prevent agent failure
        return f"Unable to search for transport options using GPTResearcher. Error: {str(ex)}. Please try using web_search instead."


@tool("accommodation_search", description="A tool for searching various accommodation options including hotels, resorts, homestays, and vacation rentals.")
async def accommodation_search(query: str) -> str:
    """
    Search the web for various accommodation options based on the given query.
    Returns a detailed research report with accommodation options, pricing, amenities, and booking information.
    
    Args:
        query: Search query for accommodation options (e.g., "hotels in Goa under 5000 per night December 2024")
    
    Returns:
        str: Research report with accommodation options and sources
    """
    try:
        # Initialize GPTResearcher with query and report type
        import os
        
        # Set environment variables for GPTResearcher configuration
        os.environ['RETRIEVER'] = 'duckduckgo'
        os.environ['SCRAPER'] = 'zendriver'
        
        researcher = GPTResearcher(
            query=query, 
            report_type='research_report',
            config_path=None,
            role="Travel agent helping to find accommodation options"
        )
        
        print(f"Accommodation search - Using retriever: duckduckgo")
        print(f"Accommodation search - Using scraper: zendriver")
        
        # Conduct research
        await researcher.conduct_research()
        
        # Generate report
        report = await researcher.write_report()
        
        # Get additional context (optional - can be used for enhanced responses)
        research_sources = researcher.get_research_sources()
        research_images = researcher.get_research_images()
        
        print(f"Accommodation search - Found {len(research_sources)} sources, {len(research_images)} images")
        
        # Format response with report, sources, and images
        formatted_response = f"{report}\n\n"
        
        # Add sources if available
        if research_sources:
            formatted_response += f"Sources: {', '.join(research_sources[:5])}\n\n"
        
        # Add images if available
        if research_images:
            formatted_response += f"Images: {', '.join(research_images[:3])}"
        
        return formatted_response
        
    except Exception as ex:
        tb_str = traceback.format_exc()
        print(f"Accommodation search error: {ex}")
        print(tb_str)
        # Return error message instead of raising to prevent agent failure
        return f"Unable to search for accommodation options using GPTResearcher. Error: {str(ex)}. Please try using web_search instead."


def internet_search(query: str, max_results: int = 3):
    """Search the web for information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results,timeout=30,safesearch='off', timelimit='y',region='us-en',backend="duckduckgo"))
            if not results:
                return "No results found."

            # Format results as a list of dictionaries
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                    "snippet": result.get("body", "")
                })
            return str(formatted_results)
    except DDGSException as e:
        print(f"DuckDuckGo search error: {str(e)}")
        return "No results found. Please try a different search query."
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return f"Search failed: {str(e)}"