use std::{ops::AddAssign, pin::Pin};

use async_trait::async_trait;
use bytes::{Buf, BytesMut};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};

use crate::{
    Error, Llm, LlmResponse, LlmTier, LlmUsage,
    parser::{LlmProtocol, ResponseStream, StreamEvent},
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

#[derive(Default)]
pub struct GeminiProtocol {
    pending_usage: Option<LlmUsage>,
}

impl LlmProtocol for GeminiProtocol {
    fn decode(&mut self, buffer: &mut BytesMut) -> Result<Option<StreamEvent>, Error> {
        if let Some(usage) = self.pending_usage.take() {
            return Ok(Some(StreamEvent::Usage(usage)));
        }

        if buffer.is_empty() {
            return Ok(None);
        }

        let mut deserializer =
            serde_json::Deserializer::from_slice(buffer).into_iter::<Vec<GeminiApiResponse>>();

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
                    usage.completion_tokens += response.usage_metadata.candidates_token_count;
                    usage.cached_tokens += response.usage_metadata.cached_content_token_count;
                }

                buffer.advance(offset);

                if usage.prompt_tokens > 0 || usage.completion_tokens > 0 {
                    self.pending_usage = Some(usage);
                }

                if !text.is_empty() {
                    Ok(Some(StreamEvent::Text(text)))
                } else if let Some(usage) = self.pending_usage.take() {
                    Ok(Some(StreamEvent::Usage(usage)))
                } else {
                    Ok(None)
                }
            }
            Some(Err(e)) if e.is_eof() => Ok(None),
            Some(Err(e)) => {
                buffer.clear();
                Err(e.into())
            }
            None => Ok(None),
        }
    }
}

/// Gemini LLM interface
///
/// # Usage
///
/// ```rust
/// let client = Client::new();
/// let gemini_api_key = std::env::var("GEMINI_API_KEY").unwrap();
/// let llm = GeminiLlm::new(client, gemini_api_key, None);
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

        let stream =
            ResponseStream::new(Box::pin(response.bytes_stream()), GeminiProtocol::default());

        Ok(Box::pin(stream))
    }
}
