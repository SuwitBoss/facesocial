"""
Database connections and session management - FIXED VERSION
Handles PostgreSQL and Redis connections with proper error handling
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    async_sessionmaker, 
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager for PostgreSQL and Redis with improved error handling"""
    
    def __init__(self):
        self.postgres_engine: Optional[AsyncEngine] = None
        self.postgres_session_factory: Optional[async_sessionmaker] = None
        self.redis_client: Optional[Redis] = None
        self._initialized = False
        self._postgres_connection_pool_size = 5
        self._postgres_max_overflow = 10
        self._redis_connection_pool_size = 10
    
    async def initialize(self):
        """Initialize database connections with comprehensive error handling"""
        if self._initialized:
            logger.info("Database manager already initialized")
            return
        
        try:
            # Initialize PostgreSQL
            await self._init_postgres()
            
            # Initialize Redis
            await self._init_redis()
            
            self._initialized = True
            logger.info("✅ Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connections: {e}")
            await self._cleanup_partial_initialization()
            raise
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection with retry logic"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing PostgreSQL connection (attempt {attempt + 1}/{max_retries})")
                
                # Create async engine with optimized settings
                self.postgres_engine = create_async_engine(
                    settings.POSTGRES_URL,
                    echo=settings.DEBUG,
                    poolclass=NullPool if settings.DEBUG else None,
                    pool_pre_ping=True,
                    pool_recycle=3600,  # 1 hour
                    pool_size=self._postgres_connection_pool_size,
                    max_overflow=self._postgres_max_overflow,
                    pool_timeout=30,
                    connect_args={
                        "server_settings": {
                            "application_name": settings.SERVICE_NAME,
                            "jit": "off"  # Disable JIT for better performance
                        },
                        "command_timeout": 30,
                        "statement_timeout": 300  # 5 minutes for long queries
                    }
                )
                
                # Create session factory
                self.postgres_session_factory = async_sessionmaker(
                    bind=self.postgres_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=True,
                    autocommit=False
                )
                
                # Test connection
                async with self.postgres_engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1 as test"))
                    test_value = result.scalar()
                    if test_value != 1:
                        raise Exception("PostgreSQL connection test failed")
                
                logger.info("✅ PostgreSQL connection established successfully")
                return
                
            except Exception as e:
                logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay)
    
    async def _init_redis(self):
        """Initialize Redis connection with retry logic"""
        max_retries = 3
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing Redis connection (attempt {attempt + 1}/{max_retries})")
                
                # Parse Redis URL and create client
                redis_url = settings.REDIS_URL
                
                # Create Redis client with connection pool
                self.redis_client = redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=10,
                    socket_connect_timeout=10,
                    retry_on_timeout=True,
                    health_check_interval=30,
                    max_connections=self._redis_connection_pool_size,
                    retry_on_error=[RedisConnectionError]
                )
                
                # Test connection
                pong = await self.redis_client.ping()
                if not pong:
                    raise Exception("Redis ping failed")
                
                # Test basic operations
                await self.redis_client.set("test_key", "test_value", ex=5)
                test_value = await self.redis_client.get("test_key")
                if test_value != "test_value":
                    raise Exception("Redis operation test failed")
                await self.redis_client.delete("test_key")
                
                logger.info("✅ Redis connection established successfully")
                return
                
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay)
    
    @asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get PostgreSQL session with automatic cleanup and error handling"""
        if not self._initialized:
            await self.initialize()
        
        if not self.postgres_session_factory:
            raise RuntimeError("PostgreSQL not initialized")
        
        session = None
        try:
            session = self.postgres_session_factory()
            yield session
            await session.commit()
            
        except SQLAlchemyError as e:
            if session:
                await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
            
        except Exception as e:
            if session:
                await session.rollback()
            logger.error(f"Unexpected error in database session: {e}")
            raise
            
        finally:
            if session:
                await session.close()
    
    async def get_redis_client(self) -> Redis:
        """Get Redis client with connection validation"""
        if not self._initialized:
            await self.initialize()
        
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        # Validate connection is alive
        try:
            await self.redis_client.ping()
        except RedisError as e:
            logger.warning(f"Redis connection validation failed: {e}")
            # Try to reconnect
            await self._init_redis()
        
        return self.redis_client
    
    async def _cleanup_partial_initialization(self):
        """Clean up partial initialization in case of failure"""
        try:
            if self.postgres_engine:
                await self.postgres_engine.dispose()
                self.postgres_engine = None
                self.postgres_session_factory = None
            
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def close(self):
        """Clean shutdown of all database connections"""
        try:
            logger.info("Shutting down database connections...")
            
            # Close PostgreSQL
            if self.postgres_engine:
                await self.postgres_engine.dispose()
                self.postgres_engine = None
                self.postgres_session_factory = None
                logger.info("PostgreSQL connections closed")
            
            # Close Redis
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
                logger.info("Redis connections closed")
            
            self._initialized = False
            logger.info("✅ Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all database connections"""
        health = {
            "postgres": {"status": False, "error": None, "response_time_ms": None},
            "redis": {"status": False, "error": None, "response_time_ms": None},
            "timestamp": asyncio.get_event_loop().time(),
            "overall_healthy": False
        }
        
        # Check PostgreSQL
        postgres_start = asyncio.get_event_loop().time()
        try:
            if self.postgres_engine:
                async with self.postgres_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                health["postgres"]["status"] = True
                health["postgres"]["response_time_ms"] = int((asyncio.get_event_loop().time() - postgres_start) * 1000)
        except Exception as e:
            health["postgres"]["error"] = str(e)
            logger.warning(f"PostgreSQL health check failed: {e}")
        
        # Check Redis
        redis_start = asyncio.get_event_loop().time()
        try:
            if self.redis_client:
                await self.redis_client.ping()
                health["redis"]["status"] = True
                health["redis"]["response_time_ms"] = int((asyncio.get_event_loop().time() - redis_start) * 1000)
        except Exception as e:
            health["redis"]["error"] = str(e)
            logger.warning(f"Redis health check failed: {e}")
        
        # Overall health
        health["overall_healthy"] = health["postgres"]["status"] and health["redis"]["status"]
        
        return health
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics"""
        stats = {
            "postgres": {
                "pool_size": 0,
                "checked_in": 0,
                "checked_out": 0,
                "overflow": 0,
                "invalid": 0
            },
            "redis": {
                "connection_pool_created": 0,
                "connection_pool_available": 0
            }
        }
        
        try:
            if self.postgres_engine and hasattr(self.postgres_engine.pool, 'size'):
                pool = self.postgres_engine.pool
                stats["postgres"]["pool_size"] = pool.size()
                stats["postgres"]["checked_in"] = pool.checkedin()
                stats["postgres"]["checked_out"] = pool.checkedout()
                stats["postgres"]["overflow"] = pool.overflow()
                stats["postgres"]["invalid"] = pool.invalidated()
                
        except Exception as e:
            logger.debug(f"Could not get PostgreSQL pool stats: {e}")
        
        try:
            if self.redis_client and hasattr(self.redis_client.connection_pool, 'created_connections'):
                pool = self.redis_client.connection_pool
                stats["redis"]["connection_pool_created"] = pool.created_connections
                stats["redis"]["connection_pool_available"] = len(pool._available_connections)
                
        except Exception as e:
            logger.debug(f"Could not get Redis pool stats: {e}")
        
        return stats


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


# Cache utilities with enhanced error handling
class CacheManager:
    """Redis cache management utilities with error handling and fallback"""
    
    def __init__(self):
        self.default_ttl = settings.CACHE_TTL_SECONDS
        self._fallback_cache = {}  # In-memory fallback cache
        self._fallback_enabled = True
    
    async def get_redis(self) -> Redis:
        """Get Redis client with error handling"""
        try:
            return await db_manager.get_redis_client()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            if not self._fallback_enabled:
                raise
            return None
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache with fallback to in-memory cache"""
        try:
            redis_client = await self.get_redis()
            if redis_client:
                value = await redis_client.get(key)
                if value is not None:
                    return value
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
        
        # Fallback to in-memory cache
        if self._fallback_enabled and key in self._fallback_cache:
            entry = self._fallback_cache[key]
            if entry['expires_at'] > asyncio.get_event_loop().time():
                return entry['value']
            else:
                del self._fallback_cache[key]
        
        return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache with fallback to in-memory cache"""
        ttl = ttl or self.default_ttl
        
        try:
            redis_client = await self.get_redis()
            if redis_client:
                result = await redis_client.setex(key, ttl, value)
                if result:
                    return True
        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
        
        # Fallback to in-memory cache
        if self._fallback_enabled:
            self._fallback_cache[key] = {
                'value': value,
                'expires_at': asyncio.get_event_loop().time() + ttl
            }
            # Clean up expired entries
            await self._cleanup_fallback_cache()
            return True
        
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        deleted = False
        
        try:
            redis_client = await self.get_redis()
            if redis_client:
                deleted = bool(await redis_client.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete failed for key {key}: {e}")
        
        # Also delete from fallback cache
        if key in self._fallback_cache:
            del self._fallback_cache[key]
            deleted = True
        
        return deleted
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await self.get_redis()
            if redis_client:
                return bool(await redis_client.exists(key))
        except Exception as e:
            logger.warning(f"Redis exists check failed for key {key}: {e}")
        
        # Check fallback cache
        if key in self._fallback_cache:
            entry = self._fallback_cache[key]
            if entry['expires_at'] > asyncio.get_event_loop().time():
                return True
            else:
                del self._fallback_cache[key]
        
        return False
    
    async def _cleanup_fallback_cache(self):
        """Clean up expired entries from fallback cache"""
        current_time = asyncio.get_event_loop().time()
        expired_keys = [
            key for key, entry in self._fallback_cache.items() 
            if entry['expires_at'] <= current_time
        ]
        for key in expired_keys:
            del self._fallback_cache[key]
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments"""
        import hashlib
        
        # Create hash from arguments
        key_data = f"{prefix}:" + ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"face_detection:{key_hash}"


# Global cache manager instance
cache_manager = CacheManager()


# Database utilities for common operations
async def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Execute a raw SQL query with parameters"""
    async with db_manager.get_postgres_session() as session:
        result = await session.execute(text(query), params or {})
        return result


async def check_database_connectivity() -> Dict[str, Any]:
    """Check database connectivity for health checks"""
    return await db_manager.health_check()


async def initialize_databases():
    """Initialize database connections (called during app startup)"""
    await db_manager.initialize()


async def close_databases():
    """Close database connections (called during app shutdown)"""
    await db_manager.close()


__all__ = [
    "DatabaseManager",
    "db_manager", 
    "get_postgres_session",
    "get_redis_client",
    "CacheManager",
    "cache_manager",
    "execute_query",
    "check_database_connectivity",
    "initialize_databases",
    "close_databases"
]