//! HTTP surface. Small set of JSON endpoints + one WS upgrade.
//!
//! Every container-creating path goes through
//! [`sandbox_guard::assert_sandbox_ready`] so we never start a shell
//! under a runtime that isn't gVisor-backed or on a network without
//! the LAN-block rules.

use axum::{
    extract::{ws::WebSocketUpgrade, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get},
    Json, Router,
};
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

use crate::{
    docker::{create_terminal, delete_terminal, list_terminals, CreateBody, TerminalHandle},
    sandbox_guard::{assert_sandbox_ready, sandbox_status},
    state::AppState,
    ws,
};

pub fn router(state: AppState) -> Router {
    // Loopback by default, but we still want the webui — served
    // potentially from a different origin in dev — to be able to
    // call us. Permissive CORS is acceptable because the only way
    // to reach us is via localhost bind.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health))
        .route("/v1/sandbox/status", get(status))
        .route("/v1/terminals", get(list).post(create))
        .route("/v1/terminals/:id", delete(remove))
        .route("/v1/terminals/:id/ws", get(ws_attach))
        .with_state(state)
        .layer(cors)
}

async fn health(State(state): State<AppState>) -> Response {
    let status = sandbox_status(state.docker(), state.network(), state.image()).await;
    match status {
        Ok(s) => (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "sandbox": s,
            })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "error", "error": err.to_string()})),
        )
            .into_response(),
    }
}

async fn status(State(state): State<AppState>) -> Response {
    match sandbox_status(state.docker(), state.network(), state.image()).await {
        Ok(s) => Json(s).into_response(),
        Err(err) => err_response(StatusCode::SERVICE_UNAVAILABLE, &err.to_string()),
    }
}

async fn list(State(state): State<AppState>) -> Response {
    match list_terminals(&state).await {
        Ok(t) => Json(json!({"terminals": t})).into_response(),
        Err(err) => err_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    }
}

async fn create(
    State(state): State<AppState>,
    body: Option<Json<CreateBody>>,
) -> Response {
    if let Err(err) =
        assert_sandbox_ready(state.docker(), state.network(), state.image()).await
    {
        return err_response(StatusCode::SERVICE_UNAVAILABLE, &err.to_string());
    }
    let body = body.map(|Json(b)| b).unwrap_or_default();
    match create_terminal(&state, body).await {
        Ok(t) => (StatusCode::CREATED, Json::<TerminalHandle>(t)).into_response(),
        Err(err) => err_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    }
}

async fn remove(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    match delete_terminal(&state, &id).await {
        Ok(()) => (StatusCode::NO_CONTENT, ()).into_response(),
        Err(err) => err_response(StatusCode::NOT_FOUND, &err.to_string()),
    }
}

async fn ws_attach(
    State(state): State<AppState>,
    Path(id): Path<String>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade.on_upgrade(move |socket| ws::bridge(state, id, socket))
}

fn err_response(code: StatusCode, message: &str) -> Response {
    (code, Json::<Value>(json!({"error": message}))).into_response()
}
