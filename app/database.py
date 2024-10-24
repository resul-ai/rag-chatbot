import sqlite3
from typing import List, Dict, Any
import json
import os

class Database:
    def __init__(self, db_path: str = "data/chatbot.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database and create tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create chat_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    sources TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            
            conn.commit()

    def save_chat_message(self, chat_id: str, query: str, response: Dict[str, Any]):
        """Save a chat message to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update or insert chat
            cursor.execute("""
                INSERT OR REPLACE INTO chats (chat_id, created_at, last_updated)
                VALUES (?, COALESCE((SELECT created_at FROM chats WHERE chat_id = ?), ?), ?)
            """, (chat_id, chat_id, response['timestamp'], response['timestamp']))
            
            # Save message
            cursor.execute("""
                INSERT INTO chat_history (chat_id, query, response, sources, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chat_id,
                query,
                response['response'],
                json.dumps(response.get('sources', [])),
                response['timestamp']
            ))
            
            conn.commit()

    def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a specific chat ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query, response, sources, timestamp
                FROM chat_history
                WHERE chat_id = ?
                ORDER BY timestamp ASC
            """, (chat_id,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'query': row['query'],
                    'response': row['response'],
                    'sources': json.loads(row['sources']),
                    'timestamp': row['timestamp']
                })
            
            return history

    def get_all_chats(self) -> List[Dict[str, Any]]:
        """Get all chat sessions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT chat_id, created_at, last_updated
                FROM chats
                ORDER BY last_updated DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]

    def clear_chat_history(self, chat_id: str):
        """Clear chat history for a specific chat ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
            cursor.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            
            conn.commit()