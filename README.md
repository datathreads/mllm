# mllm

A plug and play interface to prompt multiple LLM APIs from your rust programs

## Features

- A common `Llm` trait for seamless switching between providers.
- Multiple models to choose from: 
    - OpenAI (GPT-4o, GPT-4o-mini, etc.)
    - Anthropic Claude (Sonnet 3.5, Haiku, Opus)
    - Google Gemini (Gemini 1.5 Flash, Pro)
- Streaming Responses: Built-in support for real-time response streaming using `Server-Sent Events (SSE)`.
- Tiered Model Selection: Abstracted `Default`, `Concierge`, and `Expert` tiers to easily select appropriate model classes.
- Smart Parsing: 
    - Automatically extracts and parses `JSON` and `YAML` code blocks from LLM responses.
    - Robust usage tracking (tokens used).
- Built with async rust, supporting use with `tokio` and using `reqwest` internally for non-blocking network I/O.
