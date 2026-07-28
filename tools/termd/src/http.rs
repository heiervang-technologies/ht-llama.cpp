//! HTTP surface. Small set of JSON endpoints + one WS upgrade.
//!
//! Every container-creating path goes through
//! [`sandbox_guard::assert_sandbox_ready`] so we never start a shell
//! under a runtime that isn't gVisor-backed or on a network without
//! the LAN-block rules.
//!
//! When `AppState::auth_token` is set, every endpoint except
//! `/health` requires the token — as `Authorization: Bearer <t>` on
//! HTTP or `?token=<t>` on the WS upgrade. The WS query path exists
//! because browsers can't set arbitrary headers on `new WebSocket()`;
//! HTTP clients should prefer the header.

use axum::{
    extract::{ws::WebSocketUpgrade, Path, Query, Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
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

    // `/health` stays public so monitoring / readiness probes work
    // without a token. Everything else goes through `auth_guard`.
    let authed = Router::new()
        .route("/v1/sandbox/status", get(status))
        .route("/v1/terminals", get(list).post(create))
        .route("/v1/terminals/:id", delete(remove))
        .route("/v1/terminals/:id/ws", get(ws_attach))
        .route("/v1/terminals/:id/input", post(input))
        .route("/v1/terminals/:id/bootstrap-log", get(bootstrap_log))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_guard));

    Router::new()
        .route("/health", get(health))
        .merge(authed)
        .with_state(state)
        .layer(cors)
}

/// Checks `Authorization: Bearer <token>` on HTTP, `?token=<t>` on
/// GETs that will upgrade to WS. When `state.auth_token()` is None
/// this middleware is still in the chain but waves every request
/// through; keeps the router topology identical in both modes so
/// surprises are less likely.
async fn auth_guard(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let Some(expected) = state.auth_token() else {
        return next.run(req).await;
    };

    // Header path — preferred, used by every fetch() call.
    if let Some(value) = req.headers().get(AUTHORIZATION) {
        if let Ok(raw) = value.to_str() {
            if let Some(provided) = raw.strip_prefix("Bearer ").or_else(|| raw.strip_prefix("bearer ")) {
                if constant_time_eq(provided.as_bytes(), expected.as_bytes()) {
                    return next.run(req).await;
                }
            }
        }
    }

    // Query-string path — only used by WebSocket upgrades where the
    // browser can't set the Authorization header.
    if let Some(q) = req.uri().query() {
        if let Some(provided) = extract_token(q) {
            if constant_time_eq(provided.as_bytes(), expected.as_bytes()) {
                return next.run(req).await;
            }
        }
    }

    err_response(StatusCode::UNAUTHORIZED, "missing or invalid bearer token")
}

fn extract_token(query: &str) -> Option<String> {
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        let key = it.next()?;
        if key != "token" {
            continue;
        }
        let value = it.next().unwrap_or("");
        // Minimal percent-decode; we only care about the common
        // safe chars the webui will send (base64url / uuid).
        return Some(
            value
                .replace('+', " ")
                .split('%')
                .enumerate()
                .map(|(i, part)| {
                    if i == 0 {
                        part.to_string()
                    } else if part.len() >= 2 {
                        let hex = &part[..2];
                        match u8::from_str_radix(hex, 16) {
                            Ok(b) => {
                                let mut s = String::from_utf8_lossy(&[b]).into_owned();
                                s.push_str(&part[2..]);
                                s
                            }
                            Err(_) => part.to_string(),
                        }
                    } else {
                        part.to_string()
                    }
                })
                .collect::<String>(),
        );
    }
    None
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc: u8 = 0;
    for i in 0..a.len() {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}

async fn health(State(state): State<AppState>) -> Response {
    let status = sandbox_status(state.docker(), state.network(), state.image()).await;
    match status {
        Ok(s) => (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "sandbox": s,
                "auth_required": state.auth_token().is_some(),
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
    Query(_params): Query<HashMap<String, String>>,
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

#[cfg(test)]
mod tests {
    //! Pure-logic tests for the auth-adjacent helpers. The full
    //! `auth_guard` middleware integrates axum + tower + tokio and
    //! wants a live router to exercise; we cover it implicitly via
    //! the http-level smoke tests in CI and keep this module focused
    //! on the two pieces most likely to regress silently:
    //!
    //!   - `constant_time_eq` — comparison correctness (timing-
    //!     invariance is an implementation property we don't assert
    //!     directly).
    //!   - `extract_token` — query-string parsing + percent decoding,
    //!     which is hand-rolled to keep the `url` crate off our
    //!     dependency list.
    use super::{constant_time_eq, extract_token};

    // --- constant_time_eq -------------------------------------------------

    #[test]
    fn constant_time_eq_identical_is_true() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"ht-termd-token-12345", b"ht-termd-token-12345"));
    }

    #[test]
    fn constant_time_eq_different_length_is_false() {
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"abcd", b"abc"));
        assert!(!constant_time_eq(b"", b"x"));
    }

    #[test]
    fn constant_time_eq_same_length_different_bytes_is_false() {
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"aaa", b"aab"));
        // First-byte and last-byte mismatches both need to fail —
        // covers the worry that a broken short-circuit returned
        // `true` when the prefix matched.
        assert!(!constant_time_eq(b"xyzzz", b"yyzzz"));
        assert!(!constant_time_eq(b"zzzzx", b"zzzzy"));
    }

    // --- extract_token ----------------------------------------------------

    #[test]
    fn extract_token_returns_plain_value() {
        assert_eq!(extract_token("token=abc"), Some("abc".to_string()));
        assert_eq!(
            extract_token("other=1&token=xyz"),
            Some("xyz".to_string())
        );
        assert_eq!(
            extract_token("token=xyz&other=1"),
            Some("xyz".to_string())
        );
    }

    #[test]
    fn extract_token_missing_returns_none() {
        assert_eq!(extract_token(""), None);
        assert_eq!(extract_token("other=1"), None);
        assert_eq!(extract_token("tokenish=nope&foo=bar"), None);
    }

    #[test]
    fn extract_token_empty_value_is_empty_string() {
        // Semantically distinct from "missing token" — the caller
        // will reject the empty string because it won't match any
        // configured secret, but we preserve the distinction.
        assert_eq!(extract_token("token="), Some(String::new()));
        assert_eq!(extract_token("token=&other=1"), Some(String::new()));
    }

    #[test]
    fn extract_token_percent_decodes_safe_range() {
        // %20 → space, %3D → '='; common in base64-ish tokens.
        assert_eq!(
            extract_token("token=a%20b%3Dc"),
            Some("a b=c".to_string())
        );
    }

    #[test]
    fn extract_token_decodes_plus_as_space() {
        // application/x-www-form-urlencoded convention: `+` means
        // space. Browsers may emit this for WS query strings.
        assert_eq!(extract_token("token=hello+world"), Some("hello world".to_string()));
    }

    #[test]
    fn extract_token_handles_malformed_percent_gracefully() {
        // Trailing %; incomplete hex; non-hex letters. None of these
        // should panic. The decoder returns the segments verbatim
        // rather than failing, which is fine — a garbled token
        // simply won't match.
        assert!(extract_token("token=%").is_some());
        assert!(extract_token("token=%Z").is_some());
        assert!(extract_token("token=abc%GG").is_some());
    }

    #[test]
    fn extract_token_takes_first_occurrence() {
        // If someone passes `?token=a&token=b`, take the first. This
        // matches most URL parsers' behaviour and avoids a subtle
        // smuggling attack where a trailing duplicate overrides the
        // first value the server might have logged.
        assert_eq!(extract_token("token=a&token=b"), Some("a".to_string()));
    }

    #[test]
    fn extract_token_ignores_pair_without_value() {
        // `?token` alone (no `=`) isn't a real bearer token; return
        // the empty string so the caller's constant_time_eq against
        // a non-empty configured secret still rejects it.
        assert_eq!(extract_token("token"), Some(String::new()));
    }
}
