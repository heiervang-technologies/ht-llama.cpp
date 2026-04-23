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
}

impl AppState {
    pub async fn new(image: String, network: String) -> Result<Self> {
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
}
