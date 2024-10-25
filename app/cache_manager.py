import os
import json
import logging
from typing import Optional, Dict, Any
from redis import Redis
from datetime import timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class CachePrefix(Enum):
    """Cache key prefixes for different types of data"""
    QUERY_RESPONSE = "qr"      # Query-response pairs
    VECTOR_EMBEDDING = "ve"    # Vector embeddings
    DOCUMENT_CHUNK = "dc"      # Document chunks
    SEARCH_RESULT = "sr"      # Search results

class CacheManager:
    """Redis-based cache manager for RAG application"""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize Redis cache manager
        
        Args:
            ttl: Cache TTL in seconds (default: 1 hour)
        """
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        
        # Test Redis connection
        try:
            self.redis.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            
    def get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache"""
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    def set_cache(self, key: str, value: Dict[str, Any]):
        """Set value in cache"""
        try:
            self.redis.setex(
                name=key,
                time=timedelta(seconds=self.ttl),
                value=json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def generate_key(self, prefix: CachePrefix, content: str) -> str:
        """Generate cache key with type-specific prefix"""
        return f"{prefix.value}:{content}"