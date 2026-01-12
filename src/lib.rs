use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

pub mod claude;
pub mod gemini;
pub mod openai;
mod parser;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Llm error: {0}")]
    LlmUnexpected(String),
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("EventStream error: {0}")]
    EventStream(#[from] eventsource_stream::EventStreamError<reqwest::Error>),
    #[error("Json error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Default, Copy, Clone)]
pub struct LlmUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
}

impl std::ops::AddAssign for LlmUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.cached_tokens += rhs.cached_tokens;
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum LlmTier {
    #[default]
    Default,
    Concierge,
    Expert,
}

#[derive(Debug)]
pub enum ResponsePacket {
    Text(String),
    Json(JsonValue),
    Yaml(YamlValue),
    Usage(LlmUsage),
}

pub trait LlmResponse: Stream<Item = Result<ResponsePacket, Error>> + Send {}

#[async_trait]
pub trait Llm: Send + Sync {
    /// Return the model ID for a given tier.
    fn model_for_tier(&self, tier: LlmTier) -> &str;

    async fn prompt(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        tier: LlmTier,
    ) -> Result<Pin<Box<dyn LlmResponse>>, Error>;

    async fn call_content(&self, system_prompt: &str, user_prompt: &str) -> Result<String, Error> {
        let mut stream = self
            .prompt(system_prompt, user_prompt, LlmTier::Default)
            .await?;
        let mut content = String::new();

        while let Some(result) = stream.next().await {
            if let ResponsePacket::Text(t) = result? {
                content.push_str(&t);
            }
        }

        Ok(content)
    }

    async fn call_structured<T>(&self, system_prompt: &str, user_prompt: &str) -> Result<T, Error>
    where
        T: DeserializeOwned + JsonSchema + Send,
    {
        // Require JSON output
        let mut system = system_prompt.to_string();
        if !system.contains("json") {
            system.push_str("\n\nPlease respond in JSON format inside a ```json block.");
        }

        let mut stream = self.prompt(&system, user_prompt, LlmTier::Default).await?;

        let Some(Ok(ResponsePacket::Json(val))) = stream.next().await else {
            return Err(Error::LlmUnexpected(
                "Failed to parse JSON response".to_string(),
            ));
        };

        let parsed: T = serde_json::from_value(val)?;
        Ok(parsed)
    }
}
