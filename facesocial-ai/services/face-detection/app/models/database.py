"""
Database connections and session management
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    async_sessionmaker, 
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager for PostgreSQL and Redis"""
    
    def __init__(self):
        self.postgres_engine: Optional[AsyncEngine] = None
        self.postgres_session_factory: Optional[async_sessionmaker] = None
        self.redis_client: Optional[Redis] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        if self._initialized:
            return
        
        try:
            # Initialize PostgreSQL
            await self._init_postgres()
            
            # Initialize Redis
            await self._init_redis()
            
            self._initialized = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            # Create async engine
            self.postgres_engine = create_async_engine(
                settings.POSTGRES_URL,
                echo=settings.DEBUG,
                poolclass=NullPool if settings.DEBUG else None,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour
                pool_size=5,
                max_overflow=10,
                connect_args={
                    "server_settings": {
                        "application_name": settings.SERVICE_NAME,
                        "jit": "off"  # Disable JIT for better performance
                    }
                }
            )
            
            # Create session factory
            self.postgres_session_factory = async_sessionmaker(
                bind=self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Test connection
            async with self.postgres_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            # Parse Redis URL
            redis_url = settings.REDIS_URL
            
            # Create Redis client
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    @asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get PostgreSQL session with automatic cleanup"""
        if not self._initialized:
            await self.initialize()
        
        if not self.postgres_session_factory:
            raise RuntimeError("PostgreSQL not initialized")
        
        async with self.postgres_session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def get_redis_client(self) -> Redis:
        """Get Redis client"""
        if not self._initialized:
            await self.initialize()
        
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        return self.redis_client
    
    async def close(self):
        """Close all database connections"""
        try:
            # Close PostgreSQL
            if self.postgres_engine:
                await self.postgres_engine.dispose()
                self.postgres_engine = None
                self.postgres_session_factory = None
            
            # Close Redis
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
            
            self._initialized = False
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    async def health_check(self) -> dict:
        """Check health of all database connections"""
        health = {
            "postgres": False,
            "redis": False,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Check PostgreSQL
        try:
            if self.postgres_engine:
                async with self.postgres_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                health["postgres"] = True
        except Exception as e:
            logger.warning(f"PostgreSQL health check failed: {e}")
        
        # Check Redis
        try:
            if self.redis_client:
                await self.redis_client.ping()
                health["redis"] = True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
        
        return health


# Global database manager instance
db_manager = DatabaseManager()


# Dependency functions for FastAPI
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for PostgreSQL session"""
    async with db_manager.get_postgres_session() as session:
        yield session


async def get_redis_client() -> Redis:
    """FastAPI dependency for Redis client"""
    return await db_manager.get_redis_client()


# Cache utilities
class CacheManager:
    """Redis cache management utilities"""
    
    def __init__(self):
        self.default_ttl = settings.CACHE_TTL_SECONDS
    
    async def get_redis(self) -> Redis:
        """Get Redis client"""
        return await db_manager.get_redis_client()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        try:
            redis_client = await self.get_redis()
            return await redis_client.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            redis_client = await self.get_redis()
            ttl = ttl or self.default_ttl
            return await redis_client.setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            redis_client = await self.get_redis()
            return bool(await redis_client.delete(key))
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await self.get_redis()
            return bool(await redis_client.exists(key))
        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments"""
        import hashlib
        
        # Create hash from arguments
        key_data = f"{prefix}:" + ":".join(str(arg) for arg in args)
        return f"face_detection:{hashlib.md5(key_data.encode()).hexdigest()}"


# Global cache manager instance
cache_manager = CacheManager()


# Import text for SQL queries
try:
    from sqlalchemy import text
except ImportError:
    # Fallback for older SQLAlchemy versions
    from sqlalchemy.sql import text


__all__ = [
    "DatabaseManager",
    "db_manager", 
    "get_postgres_session",
    "get_redis_client",
    "CacheManager",
    "cache_manager"
]