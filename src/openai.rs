use async_trait::async_trait;
use eventsource_stream::{Event, Eventsource};
use futures::TryStreamExt;
use std::pin::Pin;

use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    Error, Llm, LlmResponse, LlmTier, LlmUsage,
    parser::{ResponseStream, StreamEvent},
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
/// ```rust,no_run
/// # use reqwest::Client;
/// # use mllm::openai::OpenAiLlm;
/// let client = Client::new();
/// let llm = OpenAiLlm::new(client, "api-key".to_string(), None);
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

async fn openai_event_parsing(event: Event) -> Result<Option<StreamEvent>, Error> {
    let data = event.data;
    if data == "[DONE]" {
        return Ok(None);
    }

    let event = serde_json::from_str::<CustomStreamEvent>(&data)
        .inspect_err(|e| {
            warn!(%data, "Failed to parse OpenAI stream chunk: {e}");
        })
        .ok(); // skip malformed events

    match event {
        Some(CustomStreamEvent::ResponseOutputTextDelta { delta }) => {
            Ok(Some(StreamEvent::Text(delta)))
        }
        Some(CustomStreamEvent::ResponseCompleted {
            response: CustomResponse { usage, .. },
        }) => Ok(Some(StreamEvent::Usage(LlmUsage {
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            cached_tokens: 0,
        }))),
        _ => Ok(None),
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

        let event_stream = response
            .bytes_stream()
            .eventsource()
            .map_err(Error::from)
            .try_filter_map(openai_event_parsing);

        let stream = ResponseStream::new(Box::pin(event_stream));

        Ok(Box::pin(stream) as Pin<Box<dyn LlmResponse>>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eventsource_stream::Event;

    #[tokio::test]
    async fn test_openai_event_parsing_text_delta() {
        let event = Event {
            id: "".to_string(),
            event: "".to_string(),
            data: r#"{
                "type": "response.output_text.delta",
                "delta": "Hello"
            }"#
            .to_string(),
            retry: None,
        };
        let result = openai_event_parsing(event).await.unwrap().unwrap();
        if let StreamEvent::Text(t) = result {
            assert_eq!(t, "Hello");
        } else {
            panic!("Expected text");
        }
    }

    #[tokio::test]
    async fn test_openai_event_parsing_completed() {
        let event = Event {
            id: "".to_string(),
            event: "".to_string(),
            data: r#"{
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 20
                    }
                }
            }"#
            .to_string(),
            retry: None,
        };
        let result = openai_event_parsing(event).await.unwrap().unwrap();
        if let StreamEvent::Usage(u) = result {
            assert_eq!(u.prompt_tokens, 10);
            assert_eq!(u.completion_tokens, 20);
        } else {
            panic!("Expected usage");
        }
    }
}
