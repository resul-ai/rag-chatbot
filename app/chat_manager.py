from typing import List, Dict, Optional
from datetime import datetime
import json
import os

class ChatManager:
    def __init__(self):
        self.sessions_dir = "data/chat_sessions"
        os.makedirs(self.sessions_dir, exist_ok=True)
        
    def create_session(self) -> str:
        """Create a new chat session"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "metadata": {
                "document_count": 0,
                "last_accessed": datetime.now().isoformat()
            }
        }
        self._save_session(session_id, session_data)
        return session_id
    
    def add_message(self, session_id: str, message: Dict) -> None:
        """Add a message to the session"""
        session_data = self._load_session(session_id)
        if session_data:
            message["timestamp"] = datetime.now().isoformat()
            session_data["messages"].append(message)
            session_data["metadata"]["last_accessed"] = datetime.now().isoformat()
            self._save_session(session_id, session_data)
    
    def get_messages(self, session_id: str) -> List[Dict]:
        """Get all messages in a session"""
        session_data = self._load_session(session_id)
        return session_data.get("messages", []) if session_data else []
    
    def get_sessions(self) -> List[Dict]:
        """Get list of all sessions"""
        sessions = []
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith(".json"):
                session_id = filename[:-5]  # Remove .json
                session_data = self._load_session(session_id)
                if session_data:
                    sessions.append({
                        "session_id": session_id,
                        "created_at": session_data["created_at"],
                        "message_count": len(session_data["messages"]),
                        "last_accessed": session_data["metadata"]["last_accessed"]
                    })
        return sorted(sessions, key=lambda x: x["last_accessed"], reverse=True)
    
    def _save_session(self, session_id: str, data: Dict) -> None:
        """Save session data to file"""
        filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_session(self, session_id: str) -> Optional[Dict]:
        """Load session data from file"""
        filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None