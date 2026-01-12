use std::{collections::VecDeque, pin::Pin, task::Poll};

use bytes::{Bytes, BytesMut};
use futures::Stream;

use crate::{Error, LlmResponse, LlmUsage, ResponsePacket};

static JSON_START_MARKER: &str = "```json";
static YAML_START_MARKER: &str = "```yaml";
static CODE_END_MARKER: &str = "```";

struct SseLineParser<'a> {
    event_type: &'a str,
    data: &'a str,
}

impl<'a> SseLineParser<'a> {
    pub fn parse(msg: &'a str) -> Self {
        let mut event_type = "";
        let mut data = "";

        for line in msg.lines() {
            if let Some(e) = line.strip_prefix("event: ") {
                event_type = e.trim();
            } else if let Some(d) = line.strip_prefix("data: ") {
                data = d.trim();
            }
        }

        Self { event_type, data }
    }
}

/// State machine for the streaming parser
#[derive(Debug)]
enum ResponseState {
    /// Emitting preamble text chunks as they arrive.
    /// Contains a small buffer to handle markers split across chunks.
    StreamingPreamble(String),
    /// ` ```json ` has been seen. We are now buffering all
    /// subsequent text into the String, looking for the end marker.
    BufferingJson(String),
    /// ` ```yaml ` has been seen. We are now buffering all
    /// subsequent text into the String, looking for the end marker.
    BufferingYaml(String),
    /// The stream has finished
    Done,
}

pub enum StreamEvent {
    Text(String),
    Usage(LlmUsage),
}

pub trait LlmProtocol: Send + Unpin {
    /// Decodes a single event from the buffer.
    /// Returns Ok(Some(event)) if an event was successfully decoded.
    /// Returns Ok(None) if more data is needed.
    /// Returns Err(e) if a fatal error occurred.
    fn decode(&mut self, buffer: &mut BytesMut) -> Result<Option<StreamEvent>, Error>;
}

pub struct SseProtocol<F> {
    parse: F,
}

impl<F> SseProtocol<F>
where
    F: Fn(&str, &str) -> Option<StreamEvent> + Send + Unpin,
{
    pub fn new(parse: F) -> Self {
        Self { parse }
    }
}

impl<F> LlmProtocol for SseProtocol<F>
where
    F: Fn(&str, &str) -> Option<StreamEvent> + Send + Unpin,
{
    fn decode(&mut self, buffer: &mut BytesMut) -> Result<Option<StreamEvent>, Error> {
        const SSE_END: &[u8] = b"\n\n";
        while let Some(end_idx) = buffer.windows(SSE_END.len()).position(|w| w == SSE_END) {
            let msg_bytes = buffer.split_to(end_idx + SSE_END.len());
            let msg_str = std::str::from_utf8(&msg_bytes).unwrap_or_default();
            let sse = SseLineParser::parse(msg_str);

            if !sse.data.is_empty()
                && sse.data != "[DONE]"
                && let Some(event) = (self.parse)(sse.event_type, sse.data)
            {
                return Ok(Some(event));
            }
        }
        Ok(None)
    }
}

pub struct ResponseStream<P: LlmProtocol> {
    /// The underlying byte stream from reqwest
    stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    /// Buffer to hold partial *raw network bytes*
    byte_buffer: BytesMut,
    /// Accumulated usage metadata
    usage: LlmUsage,
    /// The current state of the streaming parser
    state: ResponseState,
    /// The backlog of responses
    backlog: VecDeque<ResponsePacket>,
    /// The protocol handler
    protocol: P,
}

impl<P: LlmProtocol> ResponseStream<P> {
    /// Add text chunk to the relevant buffer
    pub fn process_text_chunk(&mut self, text: String) {
        // Update the internal buffer based on current state
        match &mut self.state {
            ResponseState::Done => return,
            ResponseState::BufferingJson(b) | ResponseState::BufferingYaml(b) => {
                b.push_str(&text);
                return; // still inside a block; wait for more
            }
            ResponseState::StreamingPreamble(buffer) => buffer.push_str(&text),
        }

        // Scan for markers in the preamble
        while let ResponseState::StreamingPreamble(buffer) = &mut self.state {
            let json_start = buffer
                .find(JSON_START_MARKER)
                .map(|i| (i, JSON_START_MARKER, true));
            let yaml_start = buffer
                .find(YAML_START_MARKER)
                .map(|i| (i, YAML_START_MARKER, false));

            // Pick the earliest marker
            let Some((idx, marker, is_json)) = [json_start, yaml_start]
                .into_iter()
                .flatten()
                .min_by_key(|x| x.0)
            else {
                // No markers: emit safe portion of text (buffer - max marker length)
                const SAFE_LEN: usize = 7;
                if buffer.len() > SAFE_LEN {
                    let emit_until = buffer.len() - SAFE_LEN;
                    self.backlog
                        .push_back(ResponsePacket::Text(buffer[..emit_until].to_string()));
                    *buffer = buffer[emit_until..].to_string();
                }
                break;
            };

            // Found a marker: Extract preamble and transition
            let (preamble, rest) = buffer.split_at(idx);

            if !preamble.is_empty() {
                self.backlog
                    .push_back(ResponsePacket::Text(preamble.to_string()));
            }

            let remainder = rest[marker.len()..].to_string();
            self.state = if is_json {
                ResponseState::BufferingJson(remainder)
            } else {
                ResponseState::BufferingYaml(remainder)
            };

            // Check if the block ends immediately in the same chunk
            self.check_buffers();
        }
    }

    /// Check if the current buffer already contains a complete block
    pub fn check_buffers(&mut self) {
        let (packet, remainder) = match &mut self.state {
            ResponseState::BufferingJson(b) => {
                let Some(idx) = b.find(CODE_END_MARKER) else {
                    return;
                };
                let (content, rem) = b.split_at(idx);
                let rem = rem[CODE_END_MARKER.len()..].to_string();
                (
                    serde_json::from_str(content)
                        .map(ResponsePacket::Json)
                        .unwrap_or_else(|_| ResponsePacket::Text(content.to_string())),
                    rem,
                )
            }
            ResponseState::BufferingYaml(b) => {
                let Some(idx) = b.find(CODE_END_MARKER) else {
                    return;
                };
                let (content, rem) = b.split_at(idx);
                let rem = rem[CODE_END_MARKER.len()..].to_string();
                (
                    serde_yaml::from_str(content)
                        .map(ResponsePacket::Yaml)
                        .unwrap_or_else(|_| ResponsePacket::Text(content.to_string())),
                    rem,
                )
            }
            _ => return,
        };

        // Transition back to preamble with whatever was left after the block
        self.state = ResponseState::StreamingPreamble(remainder);

        self.backlog.push_back(packet);
    }

    pub fn finish_stream(&mut self) -> Result<(), Error> {
        let old_state = std::mem::replace(&mut self.state, ResponseState::Done);

        match old_state {
            ResponseState::BufferingJson(b) if !b.is_empty() => {
                return Err(Error::LlmUnexpected(format!(
                    "Stream ended mid-block. Buffer: {b}"
                )));
            }
            ResponseState::StreamingPreamble(b) if !b.is_empty() => {
                // Flush remaining text
                self.backlog.push_back(ResponsePacket::Text(b));
            }
            _ => {}
        }
        Ok(())
    }
}

impl<P: LlmProtocol> ResponseStream<P> {
    pub fn new(
        stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
        protocol: P,
    ) -> Self {
        Self {
            stream,
            byte_buffer: BytesMut::new(),
            usage: Default::default(),
            state: ResponseState::StreamingPreamble(String::new()),
            backlog: VecDeque::default(),
            protocol,
        }
    }
}

impl<P: LlmProtocol> Stream for ResponseStream<P> {
    type Item = Result<ResponsePacket, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // 1. DRAIN: If there's anything in the backlog, it's the next item. Period.
            if let Some(packet) = this.backlog.pop_front() {
                return Poll::Ready(Some(Ok(packet)));
            }

            // 2. CHECK: If we are in a buffering state, see if we can move objects to the backlog
            this.check_buffers();
            if !this.backlog.is_empty() {
                continue;
            }

            // 3. TERMINATE: Only if backlog is empty and stream is done
            if matches!(this.state, ResponseState::Done) {
                if this.usage.prompt_tokens > 0 || this.usage.completion_tokens > 0 {
                    return Poll::Ready(Some(Ok(ResponsePacket::Usage(std::mem::take(
                        &mut this.usage,
                    )))));
                }
                return Poll::Ready(None);
            }

            // 4. FILL: Get more data from the network
            match this.stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    this.byte_buffer.extend(bytes);
                    while let Some(event) = this.protocol.decode(&mut this.byte_buffer)? {
                        match event {
                            StreamEvent::Text(t) => this.process_text_chunk(t),
                            StreamEvent::Usage(u) => this.usage += u,
                        }
                    }
                }
                Poll::Ready(None) => {
                    this.finish_stream()?; // NOTE: This might push the final text into backlog
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e.into()))),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl<P: LlmProtocol> LlmResponse for ResponseStream<P> {}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn create_test_protocol() -> SseProtocol<impl Fn(&str, &str) -> Option<StreamEvent>> {
        SseProtocol::new(|event, data| {
            if event == "message" {
                Some(StreamEvent::Text(data.to_string()))
            } else if event == "usage" {
                let parts: Vec<&str> = data.split(',').collect();
                Some(StreamEvent::Usage(LlmUsage {
                    prompt_tokens: parts[0].parse().unwrap_or(0),
                    completion_tokens: parts[1].parse().unwrap_or(0),
                    cached_tokens: 0,
                }))
            } else {
                None
            }
        })
    }

    fn create_test_stream()
    -> ResponseStream<SseProtocol<impl Fn(&str, &str) -> Option<StreamEvent>>> {
        ResponseStream::new(Box::pin(stream::empty()), create_test_protocol())
    }

    #[test]
    fn test_process_text_chunk_single_json() {
        let mut stream = create_test_stream();

        stream.process_text_chunk("Hello! ```json\n{\"id\": 1}\n```".to_string());

        // Expectation: preamble in backlog, json in backlog
        assert_eq!(stream.backlog.len(), 2);

        if let Some(ResponsePacket::Text(t)) = stream.backlog.pop_front() {
            assert_eq!(t, "Hello! ");
        } else {
            panic!("Expected text preamble");
        }

        if let Some(ResponsePacket::Json(j)) = stream.backlog.pop_front() {
            assert_eq!(j["id"], 1);
        } else {
            panic!("Expected json packet");
        }

        assert!(matches!(stream.state, ResponseState::StreamingPreamble(_)));
    }

    #[test]
    fn test_process_text_chunk_multiple_markers() {
        let mut stream = create_test_stream();

        stream.process_text_chunk(
            "First: ```json\n{\"a\": 1}``` and Second: ```json\n{\"b\": 2}```".to_string(),
        );

        // Total packets expected: Text("First: "), Json({"a": 1}), Text(" and Second: "), Json({"b": 2})
        assert_eq!(stream.backlog.len(), 4);

        match stream.backlog.pop_front() {
            Some(ResponsePacket::Text(t)) => assert_eq!(t, "First: "),
            _ => panic!("Expected text"),
        }
        match stream.backlog.pop_front() {
            Some(ResponsePacket::Json(j)) => assert_eq!(j["a"], 1),
            _ => panic!("Expected json"),
        }
        match stream.backlog.pop_front() {
            Some(ResponsePacket::Text(t)) => assert_eq!(t, " and Second: "),
            _ => panic!("Expected text"),
        }
        match stream.backlog.pop_front() {
            Some(ResponsePacket::Json(j)) => assert_eq!(j["b"], 2),
            _ => panic!("Expected json"),
        }
    }

    #[test]
    fn test_sse_line_parser() {
        let input = "event: message\ndata: hello world\n";
        let parsed = SseLineParser::parse(input);
        assert_eq!(parsed.event_type, "message");
        assert_eq!(parsed.data, "hello world");

        let input = "data: just data\n";
        let parsed = SseLineParser::parse(input);
        assert_eq!(parsed.event_type, "");
        assert_eq!(parsed.data, "just data");

        let input = ": comment\nevent: ping\n";
        let parsed = SseLineParser::parse(input);
        assert_eq!(parsed.event_type, "ping");
        assert_eq!(parsed.data, "");
    }

    #[test]
    fn test_sse_protocol_decode() {
        let mut protocol = create_test_protocol();
        let mut buffer = BytesMut::from("event: message\ndata: hello\n\n");

        let event = protocol.decode(&mut buffer).unwrap().unwrap();
        if let StreamEvent::Text(t) = event {
            assert_eq!(t, "hello");
        } else {
            panic!("Expected text");
        }
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_sse_protocol_partial_decode() {
        let mut protocol = create_test_protocol();
        let mut buffer = BytesMut::from("event: message\ndata: hel");

        // Should return None as no full event is present
        assert!(protocol.decode(&mut buffer).unwrap().is_none());

        // Append rest
        buffer.extend_from_slice(b"lo\n\n");
        let event = protocol.decode(&mut buffer).unwrap().unwrap();
        if let StreamEvent::Text(t) = event {
            assert_eq!(t, "hello");
        } else {
            panic!("Expected text");
        }
    }

    #[test]
    fn test_sse_protocol_multiple_events() {
        let mut protocol = create_test_protocol();
        let mut buffer =
            BytesMut::from("event: message\ndata: one\n\nevent: message\ndata: two\n\n");

        let event1 = protocol.decode(&mut buffer).unwrap().unwrap();
        if let StreamEvent::Text(t) = event1 {
            assert_eq!(t, "one");
        } else {
            panic!("Expected text one");
        }

        let event2 = protocol.decode(&mut buffer).unwrap().unwrap();
        if let StreamEvent::Text(t) = event2 {
            assert_eq!(t, "two");
        } else {
            panic!("Expected text two");
        }

        assert!(buffer.is_empty());
    }

    #[test]
    fn test_sse_protocol_usage() {
        let mut protocol = create_test_protocol();
        let mut buffer = BytesMut::from("event: usage\ndata: 10,20\n\n");

        let event = protocol.decode(&mut buffer).unwrap().unwrap();
        if let StreamEvent::Usage(u) = event {
            assert_eq!(u.prompt_tokens, 10);
            assert_eq!(u.completion_tokens, 20);
        } else {
            panic!("Expected usage");
        }
    }
}
