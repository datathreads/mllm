use std::{ops::AddAssign, pin::Pin};

use async_trait::async_trait;
use bytes::{Buf, Bytes, BytesMut};
use futures::Stream;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};

use crate::{
    Error, Llm, LlmResponse, LlmTier, LlmUsage,
    parser::{ResponseStream, StreamEvent},
};

// Gemini Models
const GEMINI_DEFAULT_MODEL: &str = "gemini-2.5-flash";
const GEMINI_CONCIERGE_MODEL: &str = "gemini-2.5-flash-lite";
const GEMINI_EXPERT_MODEL: &str = "gemini-2.5-pro";

#[derive(Serialize)]
struct GeminiRequest<'a> {
    contents: Vec<GeminiContent<'a>>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
    #[serde(rename = "safetySettings")]
    safety_settings: Vec<GeminiSafetySetting>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent<'a>>,
}

#[derive(Serialize)]
struct GeminiSafetySetting {
    category: &'static str,
    threshold: &'static str,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

#[derive(Serialize)]
struct GeminiContent<'a> {
    parts: Vec<GeminiPart<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
}

#[derive(Serialize)]
struct GeminiPart<'a> {
    text: &'a str,
}

#[derive(Deserialize, Default, Debug, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct GeminiUsageMetadata {
    #[serde(default)]
    pub prompt_token_count: u32,
    #[serde(default)]
    pub candidates_token_count: u32,
    #[serde(default)]
    pub cached_content_token_count: u32,
    #[serde(default)]
    pub total_token_count: u32,
}

impl AddAssign for GeminiUsageMetadata {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_token_count += rhs.prompt_token_count;
        self.candidates_token_count += rhs.candidates_token_count;
        self.cached_content_token_count += rhs.cached_content_token_count;
        self.total_token_count += rhs.total_token_count;
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct GeminiApiResponse {
    candidates: Vec<GeminiCandidate>,
    usage_metadata: GeminiUsageMetadata,
}

#[derive(Deserialize, Debug)]
struct GeminiCandidate {
    content: GeminiContentResponse,
}

#[derive(Deserialize, Debug)]
struct GeminiContentResponse {
    parts: Vec<GeminiPartResponse>,
}

#[derive(Deserialize, Debug)]
struct GeminiPartResponse {
    text: String,
}

struct GeminiEventStream {
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buffer: BytesMut,
    pending_usage: Option<LlmUsage>,
}

impl GeminiEventStream {
    fn new(stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>) -> Self {
        Self {
            stream,
            buffer: BytesMut::new(),
            pending_usage: None,
        }
    }
}

impl Stream for GeminiEventStream {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            // 1. DRAIN: Check if we have pending usage to emit
            if let Some(usage) = self.pending_usage.take() {
                return std::task::Poll::Ready(Some(Ok(StreamEvent::Usage(usage))));
            }

            // 2. PROCESS: Try to parse data from the buffer
            if !self.buffer.is_empty() {
                let mut deserializer = serde_json::Deserializer::from_slice(&self.buffer)
                    .into_iter::<Vec<GeminiApiResponse>>();

                match deserializer.next() {
                    Some(Ok(api_responses)) => {
                        let offset = deserializer.byte_offset();
                        let mut text = String::new();
                        let mut usage = LlmUsage::default();

                        for response in api_responses {
                            if let Some(candidate) = response.candidates.into_iter().next()
                                && let Some(part) = candidate.content.parts.into_iter().next()
                            {
                                text.push_str(&part.text);
                            }
                            usage.prompt_tokens += response.usage_metadata.prompt_token_count;
                            usage.completion_tokens +=
                                response.usage_metadata.candidates_token_count;
                            usage.cached_tokens +=
                                response.usage_metadata.cached_content_token_count;
                        }

                        self.buffer.advance(offset);

                        if usage.prompt_tokens > 0 || usage.completion_tokens > 0 {
                            self.pending_usage = Some(usage);
                        }

                        if !text.is_empty() {
                            return std::task::Poll::Ready(Some(Ok(StreamEvent::Text(text))));
                        } else if let Some(usage) = self.pending_usage.take() {
                            return std::task::Poll::Ready(Some(Ok(StreamEvent::Usage(usage))));
                        }
                    }
                    Some(Err(e)) if !e.is_eof() => {
                        // Malformed JSON (or start of array `[` that we can't parse as Vec yet without ending `]`)
                        self.buffer.clear();
                        return std::task::Poll::Ready(Some(Err(Error::Json(e))));
                    }
                    _ => {
                        // Need more data, continue
                    }
                }
            }

            // 3. FILL: Read more bytes
            match self.stream.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                    continue; // Loop back to try parsing
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(Error::from(e))));
                }
                std::task::Poll::Ready(None) => {
                    // Check buffer one last time? Usually EOF error handles it.
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

/// Gemini LLM interface
///
/// # Usage
///
/// ```rust,no_run
/// # use reqwest::Client;
/// # use mllm::gemini::GeminiLlm;
/// let client = Client::new();
/// let llm = GeminiLlm::new(client, "api-key".to_string(), None);
/// ```
pub struct GeminiLlm {
    client: Client,
    api_key: String,
    base_url: Url,
}

impl GeminiLlm {
    pub fn new(client: Client, api_key: String, base_url: Option<Url>) -> Self {
        Self {
            client,
            api_key,
            base_url: base_url
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com".parse().unwrap()),
        }
    }
}

#[async_trait]
impl Llm for GeminiLlm {
    fn model_for_tier(&self, tier: LlmTier) -> &str {
        match tier {
            LlmTier::Default => GEMINI_DEFAULT_MODEL,
            LlmTier::Concierge => GEMINI_CONCIERGE_MODEL,
            LlmTier::Expert => GEMINI_EXPERT_MODEL,
        }
    }

    async fn prompt(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        tier: LlmTier,
    ) -> Result<Pin<Box<dyn LlmResponse>>, Error> {
        let model = self.model_for_tier(tier);
        let contents = vec![GeminiContent {
            role: Some("user"),
            parts: vec![GeminiPart { text: user_prompt }],
        }];

        let system_instruction = if !system_prompt.is_empty() {
            Some(GeminiContent {
                role: None,
                parts: vec![GeminiPart {
                    text: system_prompt,
                }],
            })
        } else {
            None
        };

        let request_body = GeminiRequest {
            contents,
            generation_config: GeminiGenerationConfig {
                max_output_tokens: 4000,
            },
            safety_settings: vec![],
            system_instruction,
        };

        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?key={}",
            self.base_url, model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await?
            .error_for_status()?;

        let adapter = GeminiEventStream::new(Box::pin(response.bytes_stream()));
        let stream = ResponseStream::new(Box::pin(adapter));

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt, stream};

    use super::*;

    #[tokio::test]
    async fn test_gemini_event_stream_full_array() {
        let json = r#"[
            {
                "candidates": [{ "content": { "parts": [{ "text": "Hello" }] } }],
                "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5 }
            }
        ]"#;
        let stream = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(json))]);
        let mut event_stream = GeminiEventStream::new(Box::pin(stream));

        let event = event_stream.next().await.unwrap().unwrap();
        if let StreamEvent::Text(t) = event {
            assert_eq!(t, "Hello");
        } else {
            panic!("Expected text");
        }

        let event = event_stream.next().await.unwrap().unwrap();
        if let StreamEvent::Usage(u) = event {
            assert_eq!(u.prompt_tokens, 10);
            assert_eq!(u.completion_tokens, 5);
        } else {
            panic!("Expected usage");
        }

        assert!(event_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_gemini_event_stream_fragmented() {
        let part1 = r#"[ { "candidates": [ { "content": { "parts": [ { "text": "Part1" "#;
        let part2 = r#"} ] } } ] , "usageMetadata": { "promptTokenCount": 1 } } ]"#;

        let stream = stream::iter(vec![
            Ok::<_, reqwest::Error>(Bytes::from(part1)),
            Ok::<_, reqwest::Error>(Bytes::from(part2)),
        ]);
        let mut event_stream = GeminiEventStream::new(Box::pin(stream));

        let event = event_stream.next().await.unwrap().unwrap();
        if let StreamEvent::Text(t) = event {
            assert_eq!(t, "Part1");
        } else {
            panic!("Expected text");
        }

        let event = event_stream.next().await.unwrap().unwrap();
        if let StreamEvent::Usage(u) = event {
            assert_eq!(u.prompt_tokens, 1);
        } else {
            panic!("Expected usage");
        }

        assert!(event_stream.next().await.is_none());
    }
}
