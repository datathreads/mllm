use futures::StreamExt;
use mllm::{Llm, LlmTier, openai::OpenAiLlm};
use reqwest::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let openai_api_key = std::env::var("OPENAI_API_KEY").unwrap();
    // Initialize the OpenAI interface
    let llm = OpenAiLlm::new(client, openai_api_key, None);

    // Stream response
    let mut stream = llm
        .prompt(
            "You are a helpful assistant.",
            "Hello, world!",
            LlmTier::Default,
        )
        .await?;

    while let Some(result) = stream.next().await {
        match result? {
            mllm::ResponsePacket::Text(text) => print!("{}", text),
            mllm::ResponsePacket::Json(val) => println!("\nReceived JSON: {val}"),
            mllm::ResponsePacket::Yaml(val) => println!("\nReceived YAML: {val:?}"),
            mllm::ResponsePacket::Usage(usage) => println!("\nUsage: {usage:?}"),
        }
    }

    Ok(())
}
