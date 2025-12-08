//! Media Manager for handling audio file uploads and management
//!
//! The MediaManager handles:
//! - Listing available media files from the ./media directory
//! - File uploads (wav and mp3)
//! - MP3 to WAV conversion via ffmpeg
//! - Duration detection via ffprobe
//! - File deletion

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::RwLock;

/// Maximum upload size in bytes (1 GB)
const DEFAULT_MAX_UPLOAD_SIZE: u64 = 1024 * 1024 * 1024;

/// Supported media formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaFormat {
    Wav,
    Mp3,
}

impl MediaFormat {
    /// Get format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Some(MediaFormat::Wav),
            "mp3" => Some(MediaFormat::Mp3),
            _ => None,
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            MediaFormat::Wav => "wav",
            MediaFormat::Mp3 => "mp3",
        }
    }
}

/// Information about a media file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaFile {
    /// Unique identifier (filename without extension)
    pub id: String,
    /// Original filename
    pub filename: String,
    /// Full path to the file
    #[serde(skip)]
    pub path: PathBuf,
    /// Path to WAV version (may be same as path or converted)
    #[serde(skip)]
    pub wav_path: PathBuf,
    /// Original format
    pub format: MediaFormat,
    /// Duration in seconds (if known)
    pub duration_secs: Option<f32>,
    /// File size in bytes
    pub size_bytes: u64,
    /// Creation timestamp (Unix epoch seconds)
    pub created_at: u64,
}

impl MediaFile {
    /// Create MediaFile from a path
    pub async fn from_path(path: &Path) -> Result<Self> {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid filename",
            )))?
            .to_string();

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let format = MediaFormat::from_extension(extension)
            .ok_or_else(|| Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported format: {}", extension),
            )))?;

        let id = path
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or(&filename)
            .to_string();

        let metadata = tokio::fs::metadata(path).await?;
        let size_bytes = metadata.len();
        let created_at = metadata
            .created()
            .or_else(|_| metadata.modified())
            .map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs())
            .unwrap_or(0);

        // For MP3 files, we'll need to convert to WAV
        let wav_path = if format == MediaFormat::Mp3 {
            path.with_extension("wav")
        } else {
            path.to_path_buf()
        };

        // Try to get duration
        let duration_secs = get_audio_duration(path).await.ok();

        Ok(Self {
            id,
            filename,
            path: path.to_path_buf(),
            wav_path,
            format,
            duration_secs,
            size_bytes,
            created_at,
        })
    }
}

/// Get audio duration using ffprobe
pub async fn get_audio_duration(path: &Path) -> Result<f32> {
    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .await?;

    if !output.status.success() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr)),
        )));
    }

    let duration_str = String::from_utf8_lossy(&output.stdout);
    duration_str
        .trim()
        .parse()
        .map_err(|_| Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid duration: {}", duration_str),
        )))
}

/// Convert MP3 to WAV using ffmpeg
pub async fn convert_to_wav(input: &Path, output: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-i",
        ])
        .arg(input)
        .args([
            "-ar", "16000",      // 16kHz sample rate
            "-ac", "1",          // Mono
            "-f", "wav",
            "-y",                // Overwrite
        ])
        .arg(output)
        .status()
        .await?;

    if !status.success() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "ffmpeg conversion failed",
        )));
    }

    Ok(())
}

/// Media Manager configuration
#[derive(Debug, Clone)]
pub struct MediaManagerConfig {
    /// Directory for media files
    pub media_dir: PathBuf,
    /// Maximum upload size in bytes
    pub max_upload_size: u64,
}

impl Default for MediaManagerConfig {
    fn default() -> Self {
        Self {
            media_dir: PathBuf::from("./media"),
            max_upload_size: DEFAULT_MAX_UPLOAD_SIZE,
        }
    }
}

impl MediaManagerConfig {
    /// Create from environment variables
    pub fn from_env() -> Self {
        let media_dir = std::env::var("MEDIA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("./media"));

        let max_upload_size = std::env::var("MAX_UPLOAD_SIZE_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(DEFAULT_MAX_UPLOAD_SIZE);

        Self {
            media_dir,
            max_upload_size,
        }
    }
}

/// Manager for media files
pub struct MediaManager {
    config: MediaManagerConfig,
    files: RwLock<HashMap<String, MediaFile>>,
}

impl MediaManager {
    /// Create a new MediaManager
    pub fn new(config: MediaManagerConfig) -> Self {
        Self {
            config,
            files: RwLock::new(HashMap::new()),
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self::new(MediaManagerConfig::from_env())
    }

    /// Initialize the media manager (create directory if needed, scan files)
    pub async fn init(&self) -> Result<()> {
        // Create media directory if it doesn't exist
        if !self.config.media_dir.exists() {
            tokio::fs::create_dir_all(&self.config.media_dir).await?;
            eprintln!("[MediaManager] Created media directory: {}", self.config.media_dir.display());
        }

        // Scan for existing files
        self.scan().await?;

        Ok(())
    }

    /// Scan media directory for files
    pub async fn scan(&self) -> Result<()> {
        let mut files = self.files.write().await;
        files.clear();

        let mut entries = tokio::fs::read_dir(&self.config.media_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Skip directories and non-media files
            if path.is_dir() {
                continue;
            }

            let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if MediaFormat::from_extension(extension).is_none() {
                continue;
            }

            match MediaFile::from_path(&path).await {
                Ok(media_file) => {
                    eprintln!(
                        "[MediaManager] Found: {} ({:.1}s, {:.1} MB)",
                        media_file.filename,
                        media_file.duration_secs.unwrap_or(0.0),
                        media_file.size_bytes as f64 / 1024.0 / 1024.0
                    );
                    files.insert(media_file.id.clone(), media_file);
                }
                Err(e) => {
                    eprintln!("[MediaManager] Error scanning {}: {}", path.display(), e);
                }
            }
        }

        eprintln!("[MediaManager] Scanned {} media files", files.len());
        Ok(())
    }

    /// List all media files
    pub async fn list_files(&self) -> Vec<MediaFile> {
        let files = self.files.read().await;
        files.values().cloned().collect()
    }

    /// Get a media file by ID
    pub async fn get_file(&self, id: &str) -> Option<MediaFile> {
        let files = self.files.read().await;
        files.get(id).cloned()
    }

    /// Get the WAV path for a media file (converting if necessary)
    pub async fn get_wav_path(&self, id: &str) -> Result<PathBuf> {
        let file = self.get_file(id).await.ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Media file not found: {}", id),
            ))
        })?;

        // If it's already WAV, return the path
        if file.format == MediaFormat::Wav {
            return Ok(file.path);
        }

        // For MP3, check if WAV exists or convert
        if !file.wav_path.exists() {
            eprintln!("[MediaManager] Converting {} to WAV...", file.filename);
            convert_to_wav(&file.path, &file.wav_path).await?;
            eprintln!("[MediaManager] Conversion complete: {}", file.wav_path.display());
        }

        Ok(file.wav_path)
    }

    /// Upload a new media file
    pub async fn upload(&self, filename: &str, data: bytes::Bytes) -> Result<MediaFile> {
        // Check file size
        if data.len() as u64 > self.config.max_upload_size {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "File too large: {} bytes (max: {} bytes)",
                    data.len(),
                    self.config.max_upload_size
                ),
            )));
        }

        // Validate extension
        let extension = Path::new(filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if MediaFormat::from_extension(extension).is_none() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported format: {}. Use .wav or .mp3", extension),
            )));
        }

        // Generate unique filename if needed
        let sanitized_filename = sanitize_filename(filename);
        let mut target_path = self.config.media_dir.join(&sanitized_filename);

        // Avoid overwriting existing files
        let mut counter = 1;
        while target_path.exists() {
            let stem = Path::new(&sanitized_filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("file");
            let new_name = format!("{}_{}.{}", stem, counter, extension);
            target_path = self.config.media_dir.join(&new_name);
            counter += 1;
        }

        // Write the file
        tokio::fs::write(&target_path, &data).await?;
        eprintln!(
            "[MediaManager] Uploaded: {} ({:.1} MB)",
            target_path.display(),
            data.len() as f64 / 1024.0 / 1024.0
        );

        // Create MediaFile entry
        let media_file = MediaFile::from_path(&target_path).await?;

        // Add to cache
        let mut files = self.files.write().await;
        files.insert(media_file.id.clone(), media_file.clone());

        Ok(media_file)
    }

    /// Delete a media file
    pub async fn delete(&self, id: &str) -> Result<()> {
        let file = {
            let mut files = self.files.write().await;
            files.remove(id)
        };

        let file = file.ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Media file not found: {}", id),
            ))
        })?;

        // Delete the original file
        if file.path.exists() {
            tokio::fs::remove_file(&file.path).await?;
        }

        // Delete the WAV conversion if it exists and is different
        if file.wav_path != file.path && file.wav_path.exists() {
            tokio::fs::remove_file(&file.wav_path).await?;
        }

        eprintln!("[MediaManager] Deleted: {}", file.filename);
        Ok(())
    }

    /// Get the media directory path
    pub fn media_dir(&self) -> &Path {
        &self.config.media_dir
    }

    /// Get max upload size
    pub fn max_upload_size(&self) -> u64 {
        self.config.max_upload_size
    }
}

impl Default for MediaManager {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Sanitize a filename for safe storage
fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Thread-safe reference to MediaManager
pub type SharedMediaManager = Arc<MediaManager>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_format() {
        assert_eq!(MediaFormat::from_extension("wav"), Some(MediaFormat::Wav));
        assert_eq!(MediaFormat::from_extension("WAV"), Some(MediaFormat::Wav));
        assert_eq!(MediaFormat::from_extension("mp3"), Some(MediaFormat::Mp3));
        assert_eq!(MediaFormat::from_extension("ogg"), None);
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("test.wav"), "test.wav");
        assert_eq!(sanitize_filename("my file.wav"), "my_file.wav");
        assert_eq!(sanitize_filename("test<>:?.wav"), "test_____.wav");
    }
}
