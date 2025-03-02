# OpenWebUI Custom Pipelines

## Anthropic Manifold Pipeline

The Anthropic Manifold Pipeline provides access to Claude 3 models through the Anthropic API.

### Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)

### Available Models

- claude-3.5-haiku
- claude-3.5-sonnet  
- claude-3.7-sonnet

### Special Features

#### Image Support

The pipeline supports sending up to 5 images per API call, with a total size limit of 100MB. Images can be provided as either base64-encoded data or URLs.

#### Thinking Tag (claude-3.7-sonnet only)

The claude-3.7-sonnet model supports a special "thinking" mode that allows the model to show its reasoning process. This can be activated using the `<think>` or `[think]` tag in beginning of message (default thinking budget 16,000) or `<think:{budget_tokens}>` or `[think:{budget_tokens}]` to set custom budget tokens.

## Fireworks Manifold Pipeline

The Fireworks Manifold Pipeline provides access to Fireworks AI models through their API.

### Environment Variables

- `FIREWORKS_API_KEY`: Your Fireworks API key (required)

### Available Models

- llama-v3p1-405b-instruct
- deepseek-v3
- deepseek-r1

### Features

The pipeline supports standard chat completion parameters including:
- Temperature control
- Top-k and top-p sampling
- Presence and frequency penalties
- Streaming responses

## Storm Wiki Pipeline

The Storm Wiki Pipeline uses the knowledge_storm library to research topics and create Wikipedia-like content with summaries and varied perspectives.

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `YOU_API_KEY`: Your You.com API key (required)
- `REGULAR_MODEL_NAME`: Model for regular processing (default: gpt-4o-mini)
- `SMART_MODEL_NAME`: Model for advanced processing (default: gpt-4o)

### Usage

Simply provide a research topic in your first message, and the pipeline will automatically extract the topic and generate comprehensive Wikipedia-style content with citations.
