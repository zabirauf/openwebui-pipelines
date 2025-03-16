"""
title: Anthropic Manifold Pipeline
author: zabirauf
date: 2025-03-02
version: 1.1
license: MIT
description: A pipeline for generating text and processing images using the Anthropic API.
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import sseclient

from utils.pipelines.main import pop_system_message


class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""
        THINKING_BUDGET: int = 16000  # Default thinking budget

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"

        self.valves = self.Valves(
            **{
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "your-api-key-here"),
                "THINKING_BUDGET": int(os.getenv("THINKING_BUDGET", "16000"))
            }
        )
        self.url = 'https://api.anthropic.com/v1/messages'
        self.update_headers()

    def update_headers(self):
        self.headers = {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': self.valves.ANTHROPIC_API_KEY
        }

    def get_anthropic_models(self):
        return [
            # {"id": "claude-3-haiku-20240307", "name": "claude-3-haiku"},
            # {"id": "claude-3-opus-20240229", "name": "claude-3-opus"},
            # {"id": "claude-3-sonnet-20240229", "name": "claude-3-sonnet"},
            {"id": "claude-3-5-haiku-20241022", "name": "claude-3.5-haiku"},
            {"id": "claude-3-5-sonnet-20241022", "name": "claude-3.5-sonnet"},
            {"id": "claude-3-7-sonnet-20250219", "name": "claude-3.7-sonnet"},
            {"id": "claude-3-7-sonnet-think", "name": "claude-3.7-sonnet-think"},
        ]

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        self.update_headers()

    def pipelines(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_image(self, image_data):
        if image_data["url"].startswith("data:image"):
            mime_type, base64_data = image_data["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["url"]},
            }

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Store original message before any modifications
            original_message = user_message

            # Remove unnecessary keys
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            # Process thinking for models
            thinking_enabled = False
            thinking_budget = self.valves.THINKING_BUDGET  # Use value from Valves
            
            # Handle the "think" model variant
            if model_id == "claude-3-7-sonnet-think":
                # Always enable thinking for the think model variant
                thinking_enabled = True
                # Use the actual model ID for API calls
                model_id = "claude-3-7-sonnet-20250219"
            
            # Update the user's message in the messages list
            for i, message in enumerate(messages):
                if (message.get("role") == "user" and 
                    isinstance(message.get("content"), str) and
                    message.get("content") == original_message):
                    messages[i]["content"] = user_message
                    break

            system_message, messages = pop_system_message(messages)

            processed_messages = []
            image_count = 0
            total_image_size = 0

            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image_url":
                            if image_count >= 5:
                                raise ValueError("Maximum of 5 images per API call exceeded")

                            processed_image = self.process_image(item["image_url"])
                            processed_content.append(processed_image)

                            if processed_image["source"]["type"] == "base64":
                                image_size = len(processed_image["source"]["data"]) * 3 / 4
                            else:
                                image_size = 0

                            total_image_size += image_size
                            if total_image_size > 100 * 1024 * 1024:
                                raise ValueError("Total size of images exceeds 100 MB limit")

                            image_count += 1
                else:
                    processed_content = [{"type": "text", "text": message.get("content", "")}]

                processed_messages.append({"role": message["role"], "content": processed_content})

            # Prepare the payload
            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.8),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.9),
                "stop_sequences": body.get("stop", []),
                **({"system": str(system_message)} if system_message else {}),
                "stream": body.get("stream", False),
            }
            
            # Add thinking parameters if enabled
            if thinking_enabled:
                payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

                # When thinking is enabled, unset top_k and top_p parameters
                # as they're not compatible with thinking mode and set temperature to 1.0
                payload["temperature"] = 1.0
                del payload["top_k"]
                del payload["top_p"]
                
                # Ensure max_tokens is greater than thinking budget
                if payload["max_tokens"] <= thinking_budget:
                    payload["max_tokens"] = thinking_budget + 1000  # Add a buffer

            if body.get("stream", False):
                return self.stream_response(payload)
            else:
                return self.get_completion(payload)
        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, payload: dict) -> Generator:
        response = requests.post(self.url, headers=self.headers, json=payload, stream=True)

        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            current_block_type = None
            
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    
                    # Handle block start
                    if data["type"] == "content_block_start":
                        block_type = data["content_block"].get("type")
                        current_block_type = block_type
                        
                        # Add opening tag for thinking blocks
                        if block_type == "thinking":
                            yield "<think>"
                            if "thinking" in data["content_block"]:
                                yield data["content_block"]["thinking"]
                        elif block_type == "text" and "text" in data["content_block"]:
                            yield data["content_block"]["text"]
                    
                    # Handle content deltas
                    elif data["type"] == "content_block_delta":
                        delta_type = data["delta"].get("type")
                        
                        if delta_type == "thinking_delta" and "thinking" in data["delta"]:
                            yield data["delta"]["thinking"]
                        elif delta_type == "text_delta" and "text" in data["delta"]:
                            yield data["delta"]["text"]
                    
                    # Handle block stop
                    elif data["type"] == "content_block_stop":
                        # Add closing tag if ending a thinking block
                        if current_block_type == "thinking":
                            yield "</think>"
                        current_block_type = None
                    
                    # End of message
                    elif data["type"] == "message_stop":
                        break
                        
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {event.data}")
                except KeyError as e:
                    print(f"Unexpected data structure: {e}")
                    print(f"Full data: {data}")
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def get_completion(self, payload: dict) -> str:
        response = requests.post(self.url, headers=self.headers, json=payload)
        if response.status_code == 200:
            res = response.json()
            
            # Process all content blocks including thinking blocks
            output = ""
            for block in res.get("content", []):
                if block.get("type") == "thinking":
                    output += f"<think>{block.get('thinking', '')}</think>"
                elif block.get("type") == "text":
                    output += block.get("text", "")
            
            return output
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")