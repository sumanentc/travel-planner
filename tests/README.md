# Tests for RoamWise v2

This directory contains the test suite for the RoamWise travel agent application.

## Setup

Install test dependencies:

```bash
# Using uv
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_graph.py
```

### Run specific test class
```bash
pytest tests/test_graph.py::TestGetTravelAgent
```

### Run specific test method
```bash
pytest tests/test_graph.py::TestGetTravelAgent::test_get_travel_agent_returns_compiled_graph
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage
```bash
pytest --cov=. --cov-report=html
```

## Test Structure

### `test_graph.py`
Tests for the LangGraph state graph implementation:

- **TestGetTravelAgent**: Tests for the `get_travel_agent()` function
  - Graph compilation and structure
  - Node and edge verification
  
- **TestTravelAgentGraphExecution**: Tests for graph execution
  - Travel query handling
  - Chit-chat responses
  - Follow-up question processing
  - Out-of-context query handling
  - Empty query handling
  
- **TestGraphStateManagement**: Tests for state management
  - State preservation through execution
  - Message handling in state

## Mocking

Tests use mocking to avoid making actual API calls to OpenAI:
- `@patch('agents.validator.validator_agent')` mocks the validator agent
- `AsyncMock` is used for async function mocking

## Async Tests

Async tests are marked with `@pytest.mark.asyncio` and use the `asyncio_mode = auto` configuration in `pytest.ini`.

## Notes

- Tests are designed to run without requiring actual API keys
- All external API calls are mocked
- Tests verify the graph structure and execution flow
