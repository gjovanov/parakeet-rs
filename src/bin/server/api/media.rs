//! API handlers for media file management

use super::models::ApiResponse;
use crate::state::AppState;
use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    Json,
};
use std::sync::Arc;

/// List media files
pub async fn list_media(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<parakeet_rs::MediaFile>>> {
    let files = state.media_manager.list_files().await;
    Json(ApiResponse::success(files))
}

/// Upload a media file
pub async fn upload_media(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<ApiResponse<parakeet_rs::MediaFile>>, StatusCode> {
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        let filename = field.file_name().map(|s| s.to_string());
        let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

        if let Some(filename) = filename {
            match state.media_manager.upload(&filename, data).await {
                Ok(file) => return Ok(Json(ApiResponse::success(file))),
                Err(e) => return Ok(Json(ApiResponse::error(e.to_string()))),
            }
        }
    }

    Ok(Json(ApiResponse::error("No file provided")))
}

/// Delete a media file
pub async fn delete_media(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<ApiResponse<()>> {
    match state.media_manager.delete(&id).await {
        Ok(_) => Json(ApiResponse::success(())),
        Err(e) => Json(ApiResponse::error(e.to_string())),
    }
}
