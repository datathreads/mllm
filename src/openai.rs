use async_trait::async_trait;
use std::pin::Pin;

use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    Error, Llm, LlmResponse, LlmTier, LlmUsage,
    parser::{ResponseStream, SseProtocol, StreamEvent},
};

// Default OpenAI Models
const OPENAI_DEFAULT_MODEL: &str = "gpt-5-chat-latest";
const OPENAI_CONCIERGE_MODEL: &str = "gpt-5-nano-2025-08-07";
const OPENAI_EXPERT_MODEL: &str = "gpt-5.1-2025-11-13";

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    input: &'a str,
    max_output_tokens: u32,
    stream: bool,
}

#[derive(Deserialize, Debug, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
struct OpenAiUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum CustomStreamEvent {
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta { delta: String },
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: CustomResponse },
    // Catch-all for other event types we don't need
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize, Debug)]
struct CustomResponse {
    usage: OpenAiUsage,
}

/// OpenAI LLM interface
///
/// # Usage
///
/// ```rust
/// let client = Client::new();
/// let openai_api_key = std::env::var("OPENAI_API_KEY").unwrap();
/// let llm = OpenAiLlm::new(client, openai_api_key, None);
/// ```
pub struct OpenAiLlm {
    client: Client,
    api_key: String,
    base_url: Url,
}

impl OpenAiLlm {
    pub fn new(client: Client, api_key: String, base_url: Option<Url>) -> Self {
        Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com".parse().unwrap()),
        }
    }
}

#[async_trait]
impl Llm for OpenAiLlm {
    fn model_for_tier(&self, tier: LlmTier) -> &str {
        match tier {
            LlmTier::Default => OPENAI_DEFAULT_MODEL,
            LlmTier::Concierge => OPENAI_CONCIERGE_MODEL,
            LlmTier::Expert => OPENAI_EXPERT_MODEL,
        }
    }

    async fn prompt(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        tier: LlmTier,
    ) -> Result<Pin<Box<dyn LlmResponse>>, Error> {
        let model = self.model_for_tier(tier);
        let input = format!("{system_prompt}\n\n{user_prompt}");
        let request_body = OpenAiRequest {
            model,
            input: &input,
            max_output_tokens: 4000,
            stream: true,
        };

        let url = format!("{}/v1/responses", self.base_url);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()
            .await?
            .error_for_status()?;

        let stream = ResponseStream::new(
            Box::pin(response.bytes_stream()),
            SseProtocol::new(|_event: &str, data: &str| {
                let event = serde_json::from_str::<CustomStreamEvent>(data)
                    .inspect_err(|e| {
                        warn!(%data, "Failed to parse OpenAI stream chunk: {e}");
                    })
                    .ok()?;
                match event {
                    CustomStreamEvent::ResponseOutputTextDelta { delta } => {
                        Some(StreamEvent::Text(delta))
                    }
                    CustomStreamEvent::ResponseCompleted {
                        response: CustomResponse { usage, .. },
                    } => Some(StreamEvent::Usage(LlmUsage {
                        prompt_tokens: usage.input_tokens,
                        completion_tokens: usage.output_tokens,
                        cached_tokens: 0,
                    })),
                    _ => None,
                }
            }),
        );

        Ok(Box::pin(stream) as Pin<Box<dyn LlmResponse>>)
    }
}
