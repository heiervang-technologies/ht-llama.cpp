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
    routing::{delete, get, post},
    Json, Router,
};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

use crate::{
    docker::{
        create_terminal, delete_terminal, list_terminals, read_bootstrap_log, CreateBody,
        TerminalHandle,
    },
    sandbox_guard::{assert_sandbox_ready, sandbox_status},
    state::AppState,
    ws,
};

pub fn router(state: AppState) -> Router {
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
        .route("/v1/terminals/:id/input", post(input))
        .route("/v1/terminals/:id/bootstrap-log", get(bootstrap_log))
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

async fn create(State(state): State<AppState>, body: Option<Json<CreateBody>>) -> Response {
    if let Err(err) = assert_sandbox_ready(state.docker(), state.network(), state.image()).await {
        return err_response(StatusCode::SERVICE_UNAVAILABLE, &err.to_string());
    }
    let body = body.map(|Json(b)| b).unwrap_or_default();
    match create_terminal(&state, body).await {
        Ok(t) => (StatusCode::CREATED, Json::<TerminalHandle>(t)).into_response(),
        Err(err) => err_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    }
}

async fn remove(State(state): State<AppState>, Path(id): Path<String>) -> Response {
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

#[derive(Deserialize)]
struct InputBody {
    /// Raw text sent verbatim to the PTY. Include `\n` (or `\r`) to
    /// terminate a line — we don't add one.
    #[serde(default)]
    text: Option<String>,
    /// Base64-encoded bytes. Mutually exclusive with `text`; use for
    /// escape-heavy payloads or binary. Either field is fine.
    #[serde(default)]
    base64: Option<String>,
    /// When true, wrap the payload so it executes as a single line —
    /// convenient for `send_keys`-style automation that wants
    /// "run this command". Adds a trailing `\n` if missing.
    #[serde(default)]
    auto_enter: bool,
}

async fn input(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<InputBody>,
) -> Response {
    let Some(session) = state.sessions().get(&id).await else {
        return err_response(
            StatusCode::NOT_FOUND,
            "terminal has no active session (attach via WS at least once to start one)",
        );
    };

    let mut bytes: Vec<u8> = if let Some(b64) = body.base64 {
        use base64::{engine::general_purpose::STANDARD, Engine};
        match STANDARD.decode(b64.trim()) {
            Ok(b) => b,
            Err(e) => return err_response(StatusCode::BAD_REQUEST, &format!("base64: {e}")),
        }
    } else if let Some(t) = body.text {
        t.into_bytes()
    } else {
        return err_response(StatusCode::BAD_REQUEST, "provide `text` or `base64`");
    };

    if body.auto_enter && !bytes.ends_with(b"\n") && !bytes.ends_with(b"\r") {
        bytes.push(b'\n');
    }

    if let Err(err) = session.send_input(Bytes::from(bytes)).await {
        return err_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string());
    }
    Json(json!({"ok": true})).into_response()
}

async fn bootstrap_log(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match read_bootstrap_log(&state, &id).await {
        Ok(text) => Json(json!({"log": text})).into_response(),
        Err(err) => err_response(StatusCode::NOT_FOUND, &err.to_string()),
    }
}

fn err_response(code: StatusCode, message: &str) -> Response {
    (code, Json::<Value>(json!({"error": message}))).into_response()
}
