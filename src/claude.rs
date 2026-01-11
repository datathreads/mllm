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
/// ```rust
/// let client = Client::new();
/// let claude_api_key = std::env::var("CLAUDE_API_KEY").unwrap();
/// let llm = ClaudeLlm::new(client, claude_api_key, None);
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
            SseProtocol::new(|event: &str, data: &str| match event {
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
            }),
        );

        Ok(Box::pin(stream) as Pin<Box<dyn LlmResponse>>)
    }
}
