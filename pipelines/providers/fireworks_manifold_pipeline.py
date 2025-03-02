"""
title: Fireworks Manifold Pipe
author: zabirauf
author_url: https://github.com/zabirauf
funding_url: https://github.com/open-webui
version: 0.1.5
"""
from pydantic import BaseModel
from typing import List, Union, Generator, Iterator
import os
import requests
import json
import logging
from utils.pipelines.main import pop_system_message

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Pipeline:
    class Valves(BaseModel):
        FIREWORKS_API_KEY: str = ""

    def __init__(self):
        self.type = "manifold"
        self.id = "fireworks"
        self.name = "fireworks/"
        self.valves = self.Valves(
            **{"FIREWORKS_API_KEY": os.getenv("FIREWORKS_API_KEY", "your-api-key-here")}
        )
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.model_mapping = {
            "llama-v3p1-405b-instruct": "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "deepseek-v3": "accounts/fireworks/models/deepseek-v3",
            "deepseek-r1": "accounts/fireworks/models/deepseek-r1"
        }
        self.update_headers()

    def update_headers(self):
        self.headers = {
            "Authorization": f"Bearer {self.valves.FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        self.update_headers()

    def get_fireworks_models(self):
        return [
            {
                "id": "llama-v3p1-405b-instruct",
                "name": "llama-v3p1-405b-instruct",
            },
            {
                "id": "deepseek-v3", 
                "name": "deepseek-v3",
            },
            {
                "id": "deepseek-r1",
                "name": "deepseek-r1",
            },
        ]

    def pipelines(self) -> List[dict]:
        return self.get_fireworks_models()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Remove unnecessary keys
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            system_message, messages = pop_system_message(messages)

            # Get the full model ID from mapping
            model_id = model_id.replace("fireworks_pipe.", "")
            full_model_id = self.model_mapping.get(model_id, model_id)

            # Prepare the payload
            payload = {
                "model": full_model_id,
                "messages": messages,
                "max_tokens": body.get("max_tokens", 16384),
                "temperature": body.get("temperature", 0.6),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 1),
                "presence_penalty": body.get("presence_penalty", 0),
                "frequency_penalty": body.get("frequency_penalty", 0),
                "stream": body.get("stream", False),
            }

            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})

            logger.debug(f"Request payload: {payload}")
            logger.debug(f"Request headers: {self.headers}")
            logger.debug(f"Request URL: {self.url}")

            if body.get("stream", False):
                return self.stream_response(payload)
            else:
                return self.get_completion(payload)
        except Exception as e:
            logger.error(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, payload: dict) -> Generator:
        with requests.post(self.url, headers=self.headers, json=payload, stream=True) as response:
            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                content = data["choices"][0]["delta"].get("content")
                                if content:
                                    logger.debug(f"Yielding content: {content}")
                                    yield content
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {line}")
                        except KeyError as e:
                            logger.error(f"Unexpected data structure: {e}")
                            logger.error(f"Full data: {data}")

    def get_completion(self, payload: dict) -> str:
        response = requests.post(self.url, headers=self.headers, json=payload)
        if response.status_code != 200:
            logger.error(f"HTTP Error {response.status_code}: {response.text}")
            raise Exception(f"HTTP Error {response.status_code}: {response.text}")

        res = response.json()
        logger.debug(f"Received response: {res}")
        return (
            res["choices"][0]["message"]["content"]
            if "choices" in res and res["choices"]
            else ""
        )