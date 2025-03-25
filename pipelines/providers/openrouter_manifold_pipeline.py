import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message

class ModelInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    context_length: Optional[int] = None
    pricing: Optional[dict] = None

class Tools:
    """Tools class required by Open-WebUI manifold system"""
    def __init__(self):
        pass

class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(default="")
        SITE_URL: str = Field(default="")
        APP_NAME: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "openrouter"
        self.name = "openrouter/"
        self.valves = self.Valves(
            **{
                "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
                "SITE_URL": os.getenv("SITE_URL", ""),
                "APP_NAME": os.getenv("APP_NAME", "")
            }
        )
        self._models_cache = None
        self._last_fetch_time = 0
        self._cache_duration = 300  # Cache models for 5 minutes

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "HTTP-Referer": self.valves.SITE_URL,
            "X-Title": self.valves.APP_NAME,
            "Content-Type": "application/json",
        }

    def fetch_openrouter_models(self) -> List[dict]:
        """
        Fetches available models from OpenRouter API.
        Returns cached results if available and not expired.
        """
        current_time = time.time()
        
        # Return cached models if they're still valid
        if self._models_cache is not None and (current_time - self._last_fetch_time) < self._cache_duration:
            return self._models_cache

        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch models: {response.status_code} - {response.text}")

            models_data = response.json()
            
            # Transform the response into the format expected by Open WebUI
            processed_models = []
            for model in models_data.get("data", []):
                model_info = {
                    "id": model.get("id", ""),
                    "name": model.get("name", "").split("/")[-1],  # Extract name after last slash
                    "description": model.get("description", ""),
                    "context_length": model.get("context_length"),
                    "pricing": {
                        "prompt": model.get("pricing", {}).get("prompt"),
                        "completion": model.get("pricing", {}).get("completion")
                    }
                }
                processed_models.append(model_info)

            # Update cache
            self._models_cache = processed_models
            self._last_fetch_time = current_time
            
            return processed_models

        except Exception as e:
            print(f"Error fetching models: {e}")
            # Return cached models if available, even if expired
            if self._models_cache is not None:
                return self._models_cache
            # Return a basic fallback list if everything fails
            return [
                {"id": "openai/gpt-3.5-turbo", "name": "gpt-3.5-turbo"},
                {"id": "openai/gpt-4", "name": "gpt-4"},
                {"id": "anthropic/claude-3.5-sonnet", "name": "claude-3.5-sonnet"},
            ]

    def pipes(self) -> List[dict]:
        """Returns the list of available models."""
        # Static list of models so I don't pollute the list
        return [
            {
                "id": "deepseek/deepseek-chat-v3-0324",
                "name": "deepseek-chat-v3-0324",
            },
        ]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])
        
        # Add system message if present
        if system_message:
            messages.insert(0, {"role": "system", "content": str(system_message)})

        payload = {
            "model": body["model"],
            "messages": messages,
            "max_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.8),
            "top_p": body.get("top_p", 0.9),
            "stop": body.get("stop", []),
            "stream": body.get("stream", False)
        }

        url = "https://openrouter.ai/api/v1/chat/completions"

        try:
            if body.get("stream", False):
                return self.stream_response(url, self._get_headers(), payload)
            else:
                return self.non_stream_response(url, self._get_headers(), payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"HTTP Error {response.status_code}: {response.text}")

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        yield choice["delta"]["content"]
                                    elif "message" in choice and "content" in choice["message"]:
                                        yield choice["message"]["content"]

                                time.sleep(0.01)  # Small delay to prevent overwhelming the client

                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return (
                res["choices"][0]["message"]["content"]
                if "choices" in res and res["choices"]
                else ""
            )
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"