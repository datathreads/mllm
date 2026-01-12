use std::pin::Pin;

use async_trait::async_trait;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    Error, Llm, LlmResponse, LlmTier, LlmUsage,
    parser::{ResponseStream, SseProtocol, StreamEvent},
};

// Claude Models
const CLAUDE_DEFAULT_MODEL: &str = "claude-sonnet-4-5-20250929";
const CLAUDE_CONCIERGE_MODEL: &str = "claude-haiku-4-5-20251001";
const CLAUDE_EXPERT_MODEL: &str = "claude-opus-4-1-20250805";

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ClaudeRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<Message>,
    system: Option<String>,
    stream: bool,
}

#[derive(Deserialize, Debug, Default, Clone, Copy)]
struct ClaudeUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

// event: message_start
#[derive(Deserialize, Debug)]
struct ClaudeMessageStart {
    message: ClaudeMessageUsage,
}

#[derive(Deserialize, Debug)]
struct ClaudeMessageUsage {
    usage: ClaudeUsage,
}

// event: content_block_delta
#[derive(Deserialize, Debug)]
struct ClaudeContentDelta {
    delta: ClaudeTextDelta,
}

#[derive(Deserialize, Debug)]
struct ClaudeTextDelta {
    #[serde(rename = "type")]
    _delta_type: String, // "text_delta"
    text: String,
}

// event: message_delta
#[derive(Deserialize, Debug)]
struct ClaudeMessageDelta {
    delta: ClaudeUsageDelta,
}

#[derive(Deserialize, Debug)]
struct ClaudeUsageDelta {
    usage: ClaudeUsage,
}

/// Claude LLM interface
///
/// # Usage
///
/// ```rust,no_run
/// # use reqwest::Client;
/// # use mllm::claude::ClaudeLlm;
/// let client = Client::new();
/// let llm = ClaudeLlm::new(client, "api-key".to_string(), None);
/// ```
pub struct ClaudeLlm {
    client: Client,
    api_key: String,
    base_url: Url,
}

impl ClaudeLlm {
    pub fn new(client: Client, api_key: String, base_url: Option<Url>) -> Self {
        Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".parse().unwrap()),
        }
    }
}

#[async_trait]
impl Llm for ClaudeLlm {
    fn model_for_tier(&self, tier: LlmTier) -> &str {
        match tier {
            LlmTier::Default => CLAUDE_DEFAULT_MODEL,
            LlmTier::Concierge => CLAUDE_CONCIERGE_MODEL,
            LlmTier::Expert => CLAUDE_EXPERT_MODEL,
        }
    }

    async fn prompt(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        tier: LlmTier,
    ) -> Result<Pin<Box<dyn LlmResponse>>, Error> {
        let model = self.model_for_tier(tier);
        let request_body = ClaudeRequest {
            model,
            max_tokens: 4000,
            messages: vec![Message {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }],
            system: if system_prompt.is_empty() {
                None
            } else {
                Some(system_prompt.to_string())
            },
            stream: true,
        };

        let url = format!("{}/v1/messages", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?
            .error_for_status()?;

        let stream = ResponseStream::new(
            Box::pin(response.bytes_stream()),
            SseProtocol::new(parse_claude_stream_event),
        );

        Ok(Box::pin(stream) as Pin<Box<dyn LlmResponse>>)
    }
}

fn parse_claude_stream_event(event: &str, data: &str) -> Option<StreamEvent> {
    match event {
        "message_start" => serde_json::from_str::<ClaudeMessageStart>(data)
            .inspect_err(|e| warn!(%data, "Failed to parse message_start: {e}"))
            .ok()
            .map(|ClaudeMessageStart { message }| {
                StreamEvent::Usage(LlmUsage {
                    prompt_tokens: message.usage.input_tokens,
                    completion_tokens: 0,
                    cached_tokens: 0,
                })
            }),
        "content_block_delta" => serde_json::from_str::<ClaudeContentDelta>(data)
            .inspect_err(|e| warn!(%data, "Failed to parse content_block_delta: {e}"))
            .ok()
            .map(|ClaudeContentDelta { delta }| StreamEvent::Text(delta.text)),
        "message_delta" => serde_json::from_str::<ClaudeMessageDelta>(data)
            .inspect_err(|e| warn!(%data, "Failed to parse message_delta: {e}"))
            .ok()
            .map(|ClaudeMessageDelta { delta }| {
                StreamEvent::Usage(LlmUsage {
                    prompt_tokens: 0,
                    completion_tokens: delta.usage.output_tokens,
                    cached_tokens: 0,
                })
            }),
        // Handled by stream ending
        "message_stop" => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_claude_content_delta() {
        let data = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": "Hello" }
        }"#;
        let event = parse_claude_stream_event("content_block_delta", data).unwrap();
        if let StreamEvent::Text(t) = event {
            assert_eq!(t, "Hello");
        } else {
            panic!("Expected text");
        }
    }

    #[test]
    fn test_parse_claude_message_start_usage() {
        let data = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3",
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": 25, "output_tokens": 1 }
            }
        }"#;
        let event = parse_claude_stream_event("message_start", data).unwrap();
        if let StreamEvent::Usage(u) = event {
            assert_eq!(u.prompt_tokens, 25);
            assert_eq!(u.completion_tokens, 0);
        } else {
            panic!("Expected usage");
        }
    }
}
