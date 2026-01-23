# Quick Start Guide for Testing

## Installation

First, install the test dependencies:

```bash
uv pip install pytest pytest-asyncio pytest-mock
```

## Run Tests

```bash
# From the project root directory
pytest tests/test_graph.py -v
```

## Expected Output

You should see output similar to:

```
tests/test_graph.py::TestGetTravelAgent::test_get_travel_agent_returns_compiled_graph PASSED
tests/test_graph.py::TestGetTravelAgent::test_graph_has_validator_node PASSED
tests/test_graph.py::TestGetTravelAgent::test_graph_structure PASSED
tests/test_graph.py::TestTravelAgentGraphExecution::test_graph_execution_with_travel_query PASSED
tests/test_graph.py::TestTravelAgentGraphExecution::test_graph_execution_with_chit_chat PASSED
tests/test_graph.py::TestTravelAgentGraphExecution::test_graph_execution_with_followup_query PASSED
tests/test_graph.py::TestTravelAgentGraphExecution::test_graph_execution_with_out_of_context_query PASSED
tests/test_graph.py::TestTravelAgentGraphExecution::test_graph_execution_with_empty_query PASSED
tests/test_graph.py::TestTravelAgentGraphExecution::test_graph_invoke_sync PASSED
tests/test_graph.py::TestGraphStateManagement::test_state_preservation PASSED
tests/test_graph.py::TestGraphStateManagement::test_messages_in_state PASSED
```

## Test Coverage

The test suite covers:
- ✅ Graph creation and compilation
- ✅ Graph structure validation
- ✅ Travel query processing
- ✅ Chit-chat handling
- ✅ Follow-up question detection
- ✅ Out-of-context query handling
- ✅ State management
- ✅ Both sync and async execution

## Troubleshooting

### Import Errors
If you see import errors, make sure you're running from the project root:
```bash
cd /Users/sumandas/Public/workspace/python/roamwise_v2
pytest tests/test_graph.py
```

### Async Warnings
If you see async warnings, ensure `pytest-asyncio` is installed:
```bash
uv pip install pytest-asyncio
```

### Missing Dependencies
Install all dev dependencies:
```bash
uv pip install -e ".[dev]"
```
