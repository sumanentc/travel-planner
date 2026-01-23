from unittest.mock import patch, AsyncMock

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from workflow import get_travel_agent
from models.agent_models import TravelInquiry


class TestGetTravelAgent:
    """Test suite for the get_travel_agent function"""
    
    def test_get_travel_agent_returns_compiled_graph(self):
        """Test that get_travel_agent returns a compiled StateGraph"""
        graph = get_travel_agent()
        
        # Verify the graph is not None
        assert graph is not None
        
        # Verify it's a compiled graph (has invoke method)
        assert hasattr(graph, 'invoke')
        assert hasattr(graph, 'ainvoke')
    
    def test_graph_has_validator_node(self):
        """Test that the graph contains the validator node"""
        graph = get_travel_agent()
        
        # Get the graph structure
        graph_dict = graph.get_graph().to_json()
        
        # Verify validator node exists
        assert 'validator' in str(graph_dict)
    
    def test_graph_structure(self):
        """Test that the graph has correct structure with START -> validator -> END"""
        graph = get_travel_agent()
        
        # Get nodes and edges
        graph_structure = graph.get_graph()
        nodes = [node.id for node in graph_structure.nodes.values()]
        
        # Verify validator node exists
        assert 'validator' in nodes
        
        # Verify START and END nodes exist
        assert '__start__' in nodes
        assert '__end__' in nodes


class TestTravelAgentGraphExecution:
    """Test suite for executing the travel agent graph"""
    
    @pytest.mark.asyncio
    async def test_graph_execution_with_travel_query(self):
        """Test graph execution with a travel-related query"""
        graph = get_travel_agent()
        
        # Mock the TravelInquiry response
        mock_inquiry = TravelInquiry(
            inquiry_type="travel",
            reply_to_user="",
            is_followup=False,
            regenerated_question=""
        )
        
        # Create initial state
        initial_state = {
            "user_query": "I want to visit Paris in December",
            "messages": [],
        }
        
        # Mock the validator to avoid actual API calls
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            # Execute the graph
            result = await graph.ainvoke(initial_state)
            
            # Verify response exists
            assert "response" in result
            assert isinstance(result["response"], TravelInquiry)
            assert result["response"].inquiry_type == "travel"
    
    @pytest.mark.asyncio
    async def test_graph_execution_with_chit_chat(self):
        """Test graph execution with chit-chat query"""
        graph = get_travel_agent()
        
        # Mock the TravelInquiry response for chit-chat
        mock_inquiry = TravelInquiry(
            inquiry_type="chit-chat",
            reply_to_user="Hello! How can I help you with your travel plans?",
            is_followup=False,
            regenerated_question=""
        )
        
        initial_state = {
            "user_query": "Hello, how are you?",
            "messages": [],
        }
        
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            result = await graph.ainvoke(initial_state)
            
            assert result["response"].inquiry_type == "chit-chat"
            assert result["response"].reply_to_user != ""
    
    @pytest.mark.asyncio
    async def test_graph_execution_with_followup_query(self):
        """Test graph execution with a follow-up query"""
        graph = get_travel_agent()
        
        # Mock the TravelInquiry response for follow-up
        mock_inquiry = TravelInquiry(
            inquiry_type="travel",
            reply_to_user="",
            is_followup=True,
            regenerated_question="What is the weather like in Paris in December?"
        )
        
        initial_state = {
            "user_query": "What about the weather there?",
            "messages": [
                HumanMessage(content="I want to visit Paris in December"),
                AIMessage(content="Paris in December is a wonderful choice!")
            ],
        }
        
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            result = await graph.ainvoke(initial_state)
            
            assert result["response"].is_followup is True
            assert result["response"].regenerated_question != ""
            assert "Paris" in result["response"].regenerated_question
    
    @pytest.mark.asyncio
    async def test_graph_execution_with_out_of_context_query(self):
        """Test graph execution with out-of-context query"""
        graph = get_travel_agent()
        
        mock_inquiry = TravelInquiry(
            inquiry_type="out-of-context",
            reply_to_user="I'm a travel assistant and can only help with travel-related questions.",
            is_followup=False,
            regenerated_question=""
        )
        
        initial_state = {
            "user_query": "What is the square root of 144?",
            "messages": [],
        }
        
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            result = await graph.ainvoke(initial_state)
            
            assert result["response"].inquiry_type == "out-of-context"
            assert "travel" in result["response"].reply_to_user.lower()
    
    @pytest.mark.asyncio
    async def test_graph_execution_with_empty_query(self):
        """Test graph execution with empty user query"""
        graph = get_travel_agent()
        
        mock_inquiry = TravelInquiry(
            inquiry_type="out-of-context",
            reply_to_user="Please provide a question.",
            is_followup=False,
            regenerated_question=""
        )
        
        initial_state = {
            "user_query": "",
            "messages": [],
        }
        
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            result = await graph.ainvoke(initial_state)
            
            assert "response" in result
    
    def test_graph_invoke_sync(self):
        """Test synchronous invocation of the graph"""
        graph = get_travel_agent()
        
        mock_inquiry = TravelInquiry(
            inquiry_type="travel",
            reply_to_user="",
            is_followup=False,
            regenerated_question=""
        )
        
        initial_state = {
            "user_query": "Best time to visit Tokyo?",
            "messages": [],
        }
        
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            # Synchronous invoke
            result = graph.invoke(initial_state)
            
            assert "response" in result


class TestGraphStateManagement:
    """Test suite for state management in the graph"""
    
    @pytest.mark.asyncio
    async def test_state_preservation(self):
        """Test that state is properly preserved through graph execution"""
        graph = get_travel_agent()
        
        mock_inquiry = TravelInquiry(
            inquiry_type="travel",
            reply_to_user="",
            is_followup=False,
            regenerated_question=""
        )
        
        initial_state = {
            "user_query": "Plan a trip to Bali",
            "messages": [HumanMessage(content="Previous message")],
            "feedback": "Great!",
        }
        
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            result = await graph.ainvoke(initial_state)
            
            # Verify original state fields are preserved
            assert result["user_query"] == "Plan a trip to Bali"
            assert result["feedback"] == "Great!"
    
    @pytest.mark.asyncio
    async def test_messages_in_state(self):
        """Test that messages are properly handled in state"""
        graph = get_travel_agent()
        
        mock_inquiry = TravelInquiry(
            inquiry_type="travel",
            reply_to_user="",
            is_followup=False,
            regenerated_question=""
        )
        
        initial_messages = [
            HumanMessage(content="First message"),
            AIMessage(content="First response")
        ]
        
        initial_state = {
            "user_query": "Tell me more",
            "messages": initial_messages,
        }
        
        with patch('agents.validator.validator_agent') as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value={"response": mock_inquiry})
            
            result = await graph.ainvoke(initial_state)
            
            # Messages should be in the result
            assert "messages" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
