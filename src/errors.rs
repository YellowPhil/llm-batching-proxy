use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum ProxyError {
    #[error("Request timed out while waiting for batch processing")]
    Timeout,
    
    #[error("Inference service error: {0}")] 
    InferenceService(String),
    
    #[error("Invalid request: {0}")]
    BadRequest(String),
    
    #[error("Internal server error: {0}")]
    Internal(String),
    
    #[error("Service unavailable")]
    ServiceUnavailable,
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            ProxyError::Timeout => (
                StatusCode::REQUEST_TIMEOUT,
                "Request timed out while waiting for batch processing",
            ),
            ProxyError::InferenceService(_) => (
                StatusCode::BAD_GATEWAY,
                "Inference service error",
            ),
            ProxyError::BadRequest(_) => (
                StatusCode::BAD_REQUEST,
                "Invalid request",
            ),
            ProxyError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
            ),
            ProxyError::ServiceUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Service temporarily unavailable",
            ),
        };

        let body = Json(json!({
            "error": {
                "code": status.as_u16(),
                "message": error_message,
                // TODO: maybe add a middleware to handle this in debug-only mode
                "detail": self.to_string(),
            }
        }));

        (status, body).into_response()
    }
}