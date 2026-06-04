//! Shared application state — the Docker client handle and the
//! sandbox configuration knobs. Cloned cheaply via `Arc` into every
//! request handler.

use anyhow::{Context, Result};
use bollard::Docker;
use std::sync::Arc;

use crate::session::SessionRegistry;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Inner>,
}

pub struct Inner {
    pub docker: Docker,
    pub image: String,
    pub network: String,
    pub sessions: SessionRegistry,
    /// Optional shared-secret token. When set, every HTTP and WS
    /// request (except `/health`) must present it as a bearer token
    /// or, for WS, as a `?token=` query param. Set to `None` for
    /// loopback-only deployments where the network already gates
    /// access.
    pub auth_token: Option<String>,
}

impl AppState {
    pub async fn new(image: String, network: String, auth_token: Option<String>) -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()
            .context("failed to connect to the local Docker daemon")?;
        // Quick reachability probe so misconfigured daemons fail at
        // startup, not on the first container create.
        docker
            .ping()
            .await
            .context("Docker daemon ping failed — is the socket reachable?")?;
        Ok(Self {
            inner: Arc::new(Inner {
                docker,
                image,
                network,
                sessions: SessionRegistry::new(),
                auth_token,
            }),
        })
    }

    pub fn docker(&self) -> &Docker {
        &self.inner.docker
    }
    pub fn image(&self) -> &str {
        &self.inner.image
    }
    pub fn network(&self) -> &str {
        &self.inner.network
    }
    pub fn sessions(&self) -> &SessionRegistry {
        &self.inner.sessions
    }
    pub fn auth_token(&self) -> Option<&str> {
        self.inner.auth_token.as_deref()
    }
}
