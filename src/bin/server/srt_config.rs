//! SRT (Secure Reliable Transport) stream configuration
//!
//! Parses SRT encoder configuration from environment variables:
//! - SRT_ENCODER_IP: IP address of the SRT encoder
//! - SRT_CHANNELS: JSON array of channels with name and port
//! - SRT_LATENCY: SRT latency in microseconds (default: 200000)
//! - SRT_RCVBUF: SRT receive buffer size (default: 2097152)

use serde::{Deserialize, Serialize};

/// A single SRT channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrtChannel {
    /// Channel display name (e.g., "ORF1")
    pub name: String,
    /// SRT port number
    pub port: String,
}

/// SRT stream information for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrtStreamInfo {
    /// Channel index (0-based)
    pub id: usize,
    /// Channel name
    pub name: String,
    /// Port number
    pub port: String,
    /// Display label (e.g., "10.84.17.100:24001 (ORF1)")
    pub display: String,
}

/// SRT configuration loaded from environment
#[derive(Debug, Clone)]
pub struct SrtConfig {
    /// Encoder IP address
    pub encoder_ip: String,
    /// List of available channels
    pub channels: Vec<SrtChannel>,
    /// SRT latency in microseconds
    pub latency: u32,
    /// SRT receive buffer size
    pub rcvbuf: u32,
}

impl SrtConfig {
    /// Load SRT configuration from environment variables
    pub fn from_env() -> Option<Self> {
        let encoder_ip = std::env::var("SRT_ENCODER_IP").ok()?;

        if encoder_ip.is_empty() {
            return None;
        }

        let channels_json = std::env::var("SRT_CHANNELS").unwrap_or_default();
        let channels: Vec<SrtChannel> = if channels_json.is_empty() {
            vec![]
        } else {
            serde_json::from_str(&channels_json).unwrap_or_else(|e| {
                eprintln!("[SRT] Failed to parse SRT_CHANNELS: {}", e);
                vec![]
            })
        };

        if channels.is_empty() {
            eprintln!("[SRT] No channels configured");
            return None;
        }

        let latency = std::env::var("SRT_LATENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);

        let rcvbuf = std::env::var("SRT_RCVBUF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2_097_152);

        eprintln!("[SRT] Configured {} channels from encoder {}", channels.len(), encoder_ip);
        for (i, ch) in channels.iter().enumerate() {
            eprintln!("[SRT]   [{}] {} -> port {}", i, ch.name, ch.port);
        }

        Some(Self {
            encoder_ip,
            channels,
            latency,
            rcvbuf,
        })
    }

    /// Get channel by index
    pub fn get_channel(&self, id: usize) -> Option<&SrtChannel> {
        self.channels.get(id)
    }

    /// Build SRT URL for a channel
    pub fn build_srt_url(&self, channel: &SrtChannel) -> String {
        format!(
            "srt://{}:{}?mode=caller&latency={}&rcvbuf={}",
            self.encoder_ip, channel.port, self.latency, self.rcvbuf
        )
    }

    /// Get all streams as API response format
    pub fn list_streams(&self) -> Vec<SrtStreamInfo> {
        self.channels
            .iter()
            .enumerate()
            .map(|(id, ch)| SrtStreamInfo {
                id,
                name: ch.name.clone(),
                port: ch.port.clone(),
                display: format!("{}:{} ({})", self.encoder_ip, ch.port, ch.name),
            })
            .collect()
    }

    /// Check if SRT is configured
    pub fn is_configured(&self) -> bool {
        !self.encoder_ip.is_empty() && !self.channels.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_url_builder() {
        let config = SrtConfig {
            encoder_ip: "10.84.17.100".to_string(),
            channels: vec![SrtChannel {
                name: "ORF1".to_string(),
                port: "24001".to_string(),
            }],
            latency: 200000,
            rcvbuf: 2097152,
        };

        let url = config.build_srt_url(&config.channels[0]);
        assert_eq!(
            url,
            "srt://10.84.17.100:24001?mode=caller&latency=200000&rcvbuf=2097152"
        );
    }

    #[test]
    fn test_list_streams() {
        let config = SrtConfig {
            encoder_ip: "10.84.17.100".to_string(),
            channels: vec![
                SrtChannel {
                    name: "ORF1".to_string(),
                    port: "24001".to_string(),
                },
                SrtChannel {
                    name: "ORF2".to_string(),
                    port: "24002".to_string(),
                },
            ],
            latency: 200000,
            rcvbuf: 2097152,
        };

        let streams = config.list_streams();
        assert_eq!(streams.len(), 2);
        assert_eq!(streams[0].id, 0);
        assert_eq!(streams[0].name, "ORF1");
        assert_eq!(streams[0].display, "10.84.17.100:24001 (ORF1)");
    }

    #[test]
    fn test_srt_url_with_custom_latency() {
        let config = SrtConfig {
            encoder_ip: "192.168.1.1".to_string(),
            channels: vec![SrtChannel {
                name: "Test".to_string(),
                port: "5000".to_string(),
            }],
            latency: 500000,
            rcvbuf: 4194304,
        };

        let url = config.build_srt_url(&config.channels[0]);
        assert!(url.contains("latency=500000"));
        assert!(url.contains("rcvbuf=4194304"));
        assert!(url.contains("192.168.1.1:5000"));
    }

    #[test]
    fn test_empty_channels() {
        let config = SrtConfig {
            encoder_ip: "10.0.0.1".to_string(),
            channels: vec![],
            latency: 200000,
            rcvbuf: 2097152,
        };

        let streams = config.list_streams();
        assert_eq!(streams.len(), 0);
    }

    #[test]
    fn test_channel_serialization() {
        let channel = SrtChannel {
            name: "ORF1".to_string(),
            port: "24001".to_string(),
        };
        let json = serde_json::to_string(&channel).unwrap();
        assert!(json.contains("\"name\":\"ORF1\""));
        assert!(json.contains("\"port\":\"24001\""));

        let parsed: SrtChannel = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "ORF1");
        assert_eq!(parsed.port, "24001");
    }

    #[test]
    fn test_stream_info_serialization() {
        let info = SrtStreamInfo {
            id: 0,
            name: "ORF1".to_string(),
            port: "24001".to_string(),
            display: "10.84.17.100:24001 (ORF1)".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: SrtStreamInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, info.id);
        assert_eq!(parsed.name, info.name);
    }

    #[test]
    fn test_many_channels() {
        let channels: Vec<SrtChannel> = (0..14).map(|i| SrtChannel {
            name: format!("CH{}", i),
            port: format!("{}", 24001 + i),
        }).collect();

        let config = SrtConfig {
            encoder_ip: "10.84.17.100".to_string(),
            channels,
            latency: 200000,
            rcvbuf: 2097152,
        };

        let streams = config.list_streams();
        assert_eq!(streams.len(), 14);
        assert_eq!(streams[13].id, 13);
    }
}
