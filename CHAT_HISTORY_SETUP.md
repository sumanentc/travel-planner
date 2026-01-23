# Chat History Setup with SQLite

This document explains the chat history persistence setup for the RoamWise Chainlit application.

## Overview

Chat history is now enabled using SQLite database storage. This allows users to:
- Access previous conversations
- Resume interrupted chats
- View conversation history in the sidebar
- Search through past interactions

## Configuration

### 1. Database Configuration (`.chainlit/config.toml`)

The following configuration has been added to enable data persistence:

```toml
[datapersistence]
# Enable data persistence to store chat history
enabled = true

# Database connection string for SQLite
database_connection_string = "sqlite+aiosqlite:///.chainlit/chat_history.db"

# Store user sessions in the database
store_user_sessions = true

# Store threads (conversations) in the database
store_threads = true

# Store steps (individual messages) in the database
store_steps = true

# Store elements (files, images, etc.) in the database
store_elements = true

# Store feedbacks in the database
store_feedbacks = true
```

### 2. Application Updates (`app.py`)

Two new callbacks have been added:

#### `@cl.on_chat_start`
- Triggered when a new conversation starts
- Initializes the thread_id for the session
- Sends a welcome message to the user

#### `@cl.on_chat_resume`
- Triggered when a user returns to a previous conversation
- Restores the thread_id from the database
- Notifies the user that the chat has been resumed

## Database Location

The SQLite database is stored at:
```
.chainlit/chat_history.db
```

This file will be automatically created when you first run the application with persistence enabled.

## Features Enabled

### 1. **Thread History Sidebar**
- Users can see all their previous conversations in the sidebar
- Click on any thread to resume that conversation

### 2. **Conversation Continuity**
- All messages, user inputs, and agent responses are persisted
- LangGraph state is maintained separately via MemorySaver (in-memory)
- Chainlit persistence handles the UI chat history

### 3. **Session Management**
- Sessions are maintained for 15 days (configurable via `user_session_timeout`)
- Automatic cleanup of expired sessions

### 4. **Search & Filter**
- Search through conversation history
- Filter by date, tags, or content

## Usage

### Starting the Application

```bash
chainlit run app.py
```

Or if using the main block:
```bash
python app.py
```

### Accessing Chat History

1. **View History**: Click the sidebar icon (☰) to see all previous conversations
2. **Resume Chat**: Click on any previous thread to continue that conversation
3. **New Chat**: Click "New Chat" button to start a fresh conversation

### Managing History

The chat history is automatically managed by Chainlit. To manually manage:

- **Clear History**: Delete the `.chainlit/chat_history.db` file
- **Backup History**: Copy the `.chainlit/chat_history.db` file to a backup location
- **Export History**: Use Chainlit's data export features (if available in your version)

## Important Notes

### Dual State Management

This application uses two separate persistence mechanisms:

1. **LangGraph State (MemorySaver)**: 
   - Stores the workflow state (entities, messages, current step)
   - In-memory only (resets on app restart)
   - Uses `thread_id` from `cl.context.session.id`

2. **Chainlit Persistence (SQLite)**:
   - Stores the UI chat history (messages, threads, user sessions)
   - Persists across app restarts
   - Enables the history sidebar and thread resumption

### Thread ID Consistency

Both systems use the same `thread_id` (`cl.context.session.id`) to ensure:
- LangGraph can maintain conversation context
- Chainlit can display the correct chat history

### Data Privacy

- Chat history is stored locally in SQLite
- No data is sent to external servers (unless you configure cloud storage)
- User environment variables can be excluded from persistence (see `persist_user_env` in config)

## Troubleshooting

### Database Locked Error
If you encounter "database is locked" errors:
```bash
# Stop the application
# Delete the database file
rm .chainlit/chat_history.db
# Restart the application
```

### Missing Chat History
If chat history doesn't appear:
1. Verify `[datapersistence] enabled = true` in config.toml
2. Check that `.chainlit/chat_history.db` exists
3. Ensure `aiosqlite` is installed: `pip install aiosqlite`

### Performance Issues
For large chat histories:
- Consider periodic cleanup of old threads
- Use database optimization tools (VACUUM)
- Monitor database file size

## Advanced Configuration

### Custom Database Location

To use a different database location:

```toml
database_connection_string = "sqlite+aiosqlite:///path/to/your/database.db"
```

### PostgreSQL (Production)

For production deployments, consider PostgreSQL:

```toml
database_connection_string = "postgresql+asyncpg://user:password@localhost/dbname"
```

Note: Requires `asyncpg` package: `pip install asyncpg`

## Dependencies

Ensure these packages are installed:

```bash
pip install chainlit aiosqlite
```

Or add to your `pyproject.toml`:
```toml
[project]
dependencies = [
    "chainlit>=2.9.0",
    "aiosqlite>=0.19.0",
]
```

## References

- [Chainlit Data Persistence Documentation](https://docs.chainlit.io/data-persistence/overview)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
