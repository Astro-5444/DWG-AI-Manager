import os
import requests
import threading
from typing import Optional, List, Dict
import time
import base64

os.environ["LOCAL_LLM_API_KEY"] = "ak_OFIKwiNCW2UcDWkLrVRhMR-tVb9SIwaGvGeGueDk1tM"


class APIClient:
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        base_url: str = "http://localhost:8000",
        model: str = "qwen-3.5vl-Q4",
        timeout: int = 160,
    ):
        """
        Initialize client for local LLM server with API key rotation support.
        
        Args:
            api_keys: Single API key or list of API keys for rotation
            base_url: Base URL of the local LLM server
            model: Model name to use (default: qwen-3vl)
            timeout: Request timeout in seconds
        """
        # Support single key or list of keys
        if isinstance(api_keys, str):
            api_keys = [api_keys]
            
        self.api_keys = api_keys or [os.getenv("LOCAL_LLM_API_KEY")]
        self.api_keys = [k for k in self.api_keys if k]  # Remove empty strings

        if not self.api_keys:
            raise ValueError(
                "No API key found. "
                "Set LOCAL_LLM_API_KEY as an environment variable or pass a list of keys."
            )

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        
        # Key rotation
        self._key_index = 0
        self._lock = threading.Lock()

    @property
    def key_count(self):
        """Returns the number of available API keys."""
        return len(self.api_keys)

    def _get_api_key(self):
        """Get the next API key in rotation (thread-safe)."""
        with self._lock:
            key = self.api_keys[self._key_index]
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            return key

    def chat(
        self,
        user_text: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 3,
        top_p: float = 0.9,
        top_k: int = 40,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        cache_prompt: bool = False,  # NEW: Enable prompt caching
        image_data: Optional[List[Dict]] = None,  # NEW: For vision support with caching
    ) -> str:
        """
        Send text to local LLM server with automatic retries and key-switching on failure.
        
        Args:
            user_text: The user's message
            system_prompt: Optional system prompt
            history: Optional conversation history
            temperature: Randomness (0.0-2.0)
            max_tokens: Maximum output length
            max_retries: Number of retry attempts on failure
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            frequency_penalty: Penalize repeated tokens
            presence_penalty: Penalize repeated topics
            cache_prompt: Enable prompt caching for faster repeated processing
            image_data: List of image data dicts for vision models (e.g., [{"data": base64_str, "id": 10}])
            
        Returns:
            The model's response text
            
        Raises:
            ValueError: If user_text is empty
            RuntimeError: If all retry attempts fail
        """
        if not user_text or not user_text.strip():
            raise ValueError("user_text cannot be empty")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        # Request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        # NEW: Add caching and image data if provided
        if cache_prompt:
            payload["cache_prompt"] = True
        
        if image_data:
            payload["image_data"] = image_data

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                api_key = self._get_api_key()
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            
            except requests.exceptions.HTTPError as e:
                last_error = e
                # Fix: e.response is falsy for 4xx/5xx status codes, must use 'is not None'
                try:
                    error_detail = e.response.json().get('detail', str(e)) if e.response is not None else str(e)
                except:
                    error_detail = str(e)
                
                if e.response is not None and e.response.status_code == 401:
                    print(f"[AUTH ERROR] Invalid or expired API key. Error: {error_detail}", flush=True)
                elif e.response is not None and e.response.status_code == 503:
                    print(f"[SERVER ERROR] Backend server unavailable. Error: {error_detail}", flush=True)
                elif e.response is not None and e.response.status_code == 400:
                    try:
                        detailed_info = e.response.json()
                    except:
                        detailed_info = e.response.text
                    print(f"[BAD REQUEST] Payload rejected: {detailed_info}", flush=True)
                else:
                    print(f"[HTTP ERROR] Status {e.response.status_code if e.response is not None else 'N/A'}: {error_detail}", flush=True)
                
                if attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[RETRY] Attempt {attempt+1} failed. Retrying in {wait_time}s with different key...", flush=True)
                    time.sleep(wait_time)
            
            except requests.exceptions.Timeout:
                last_error = requests.exceptions.Timeout("Request timeout")
                print(f"[TIMEOUT] Request took longer than {self.timeout}s", flush=True)
                
                if attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[RETRY] Attempt {attempt+1} failed. Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
            
            except requests.exceptions.ConnectionError as e:
                last_error = e
                print(f"[CONNECTION ERROR] Could not connect to server at {self.base_url}", flush=True)
                
                if attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[RETRY] Attempt {attempt+1} failed. Retrying in {wait_time}s...", flush=True)
                    time.sleep(wait_time)
            
            except Exception as e:
                last_error = e
                print(f"[ERROR] Unexpected error: {e}", flush=True)
                
                if attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)
                    print(f"[RETRY] Attempt {attempt+1} failed. Retrying in {wait_time}s with different key...", flush=True)
                    time.sleep(wait_time)

        raise RuntimeError(f"Local LLM API call failed after {max_retries + 1} attempts: {last_error}")

    def health_check(self) -> bool:
        """
        Check if the server is healthy and reachable.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False