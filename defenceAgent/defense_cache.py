import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Try importing redis
try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger("DefenseCache")

class DefenseCache:
    """
    Layer 1 Defense: Hash/Exact Match Cache.
    Supports Redis (production) and Local JSON File (dev/fallback).
    """
    def __init__(self, redis_url: Optional[str] = None, local_file: str = "defense_cache.json"):
        self.use_redis = False
        self.redis_client = None
        self.local_cache: Dict[str, Any] = {}
        # Save local cache in the same directory as this file by default if relative
        if not os.path.isabs(local_file):
            self.local_file = Path(__file__).parent / local_file
        else:
            self.local_file = Path(local_file)
        
        # 1. Try Redis if URL provided or in env
        # Env var example: redis://localhost:6379/0
        url = redis_url or os.environ.get("DEFENSE_REDIS_URL")
        if url and redis:
            try:
                self.redis_client = redis.from_url(url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                self.use_redis = True
                print(f"[DefenseCache] Connected to Redis at {url}")
            except Exception as e:
                print(f"[DefenseCache] Failed to connect to Redis: {e}. Falling back to local file.")
        
        # 2. Fallback to Local File
        if not self.use_redis:
            print(f"[DefenseCache] Using local file cache: {self.local_file}")
            self._load_local_cache()

    def _load_local_cache(self):
        if self.local_file.exists():
            try:
                text = self.local_file.read_text(encoding="utf-8")
                if text.strip():
                    self.local_cache = json.loads(text)
            except Exception as e:
                print(f"[DefenseCache] Failed to load local cache: {e}")
                self.local_cache = {}

    def _save_local_cache(self):
        if not self.use_redis:
            try:
                # Use default=str to handle non-serializable objects like Path
                self.local_file.write_text(json.dumps(self.local_cache, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            except Exception as e:
                print(f"[DefenseCache] Failed to save local cache: {e}")

    def _compute_hash(self, image_data: Union[str, bytes, Path], text: str) -> str:
        """Compute a unique hash for the (Image, Text) pair."""
        # 1. Hash Text
        # Ensure text is a string
        text_str = str(text).strip() if text is not None else ""
        text_hash = hashlib.sha256(text_str.encode("utf-8")).hexdigest()
        
        # 2. Hash Image
        img_hash = "no_image"
        if image_data:
            try:
                if isinstance(image_data, (str, Path)):
                    p = Path(image_data)
                    if p.exists() and p.is_file():
                        img_bytes = p.read_bytes()
                        img_hash = hashlib.sha256(img_bytes).hexdigest()
                    else:
                        # Maybe it's a URL or raw string?
                        img_hash = hashlib.sha256(str(image_data).encode("utf-8")).hexdigest()
                elif isinstance(image_data, bytes):
                    img_hash = hashlib.sha256(image_data).hexdigest()
            except Exception as e:
                print(f"[DefenseCache] Warning: Failed to hash image data: {e}")
                img_hash = "hash_error"
        
        return f"{img_hash}:{text_hash}"

    def get(self, image_data: Union[str, bytes, Path], text: str) -> Optional[Dict[str, Any]]:
        """Retrieve verdict from cache."""
        try:
            key = self._compute_hash(image_data, text)
            
            if self.use_redis:
                try:
                    data = self.redis_client.get(f"def_cache:{key}")
                    if data:
                        return json.loads(data)
                except Exception as e:
                    print(f"[DefenseCache] Redis get error: {e}")
            else:
                return self.local_cache.get(key)
        except Exception as e:
            print(f"[DefenseCache] Get error: {e}")
        
        return None

    def set(self, image_data: Union[str, bytes, Path], text: str, result: Dict[str, Any]):
        """Save verdict to cache."""
        try:
            key = self._compute_hash(image_data, text)
            
            if self.use_redis:
                try:
                    # Expire in 7 days (604800 seconds)
                    # Use default=str for safety
                    self.redis_client.setex(f"def_cache:{key}", 604800, json.dumps(result, default=str))
                except Exception as e:
                    print(f"[DefenseCache] Redis set error: {e}")
            else:
                self.local_cache[key] = result
                self._save_local_cache()
        except Exception as e:
            print(f"[DefenseCache] Set error: {e}")
