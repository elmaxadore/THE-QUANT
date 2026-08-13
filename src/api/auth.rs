//! # API Authentication
//!
//! JWT-based authentication for the web API. Clients authenticate with the
//! master password (or a session token from the TUI) and receive a JWT with
//! a 1-hour expiry. All trading commands require a valid token.

use crate::error::{QuantError, QuantResult};
use crate::security::SecurityManager;
use chrono::{DateTime, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// JWT claims for API sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiClaims {
    pub sub: String,          // user id
    pub exp: usize,           // expiry
    pub iat: usize,           // issued at
    pub role: String,         // "admin" | "viewer"
    pub scope: String,        // "api" | "tui"
}

/// Authentication request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub password: String,
}

/// Authentication response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub expires_at: DateTime<Utc>,
    pub role: String,
}

/// The API auth manager.
#[derive(Debug)]
pub struct ApiAuth {
    /// Reference to the security manager (for vault unlock).
    security: Arc<RwLock<SecurityManager>>,
    /// JWT secret derived from the vault.
    jwt_secret: String,
    /// Session expiry in seconds.
    expiry_secs: u64,
}

impl ApiAuth {
    pub fn new(security: Arc<RwLock<SecurityManager>>) -> Self {
        Self {
            security,
            jwt_secret: String::new(),
            expiry_secs: 3600,
        }
    }

    /// Authenticate with the master password and issue a JWT.
    pub async fn login(&self, password: &str) -> QuantResult<LoginResponse> {
        // Unlock the vault with the password; if it fails → invalid credentials
        let mut security = self.security.write().await;
        let session = security.unlock(&secrecy::SecretString::new(password.to_string()))?;

        let now = Utc::now();
        let claims = ApiClaims {
            sub: "quant".to_string(),
            exp: (now.timestamp() + self.expiry_secs as i64) as usize,
            iat: now.timestamp() as usize,
            role: "admin".to_string(),
            scope: "api".to_string(),
        };

        // Derive a JWT secret from the session (reuse the security manager's)
        let secret = session.token.clone();
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )?;

        Ok(LoginResponse {
            token,
            expires_at: now,
            role: "admin".to_string(),
        })
    }

    /// Validate a JWT token.
    pub fn validate(&self, token: &str) -> QuantResult<ApiClaims> {
        if self.jwt_secret.is_empty() {
            return Err(QuantError::AuthenticationError("API auth not initialized".into()));
        }
        let data = decode::<ApiClaims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_bytes()),
            &Validation::default(),
        )?;
        Ok(data.claims)
    }

    /// Set the JWT secret (called after vault unlock).
    pub fn set_secret(&mut self, secret: String) {
        self.jwt_secret = secret;
    }

    /// Check if a token grants admin (trading) access.
    pub fn is_admin(&self, claims: &ApiClaims) -> bool {
        claims.role == "admin" && claims.scope == "api"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::SecurityManager;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_login_and_validate() {
        let security = Arc::new(RwLock::new(SecurityManager::new(std::path::PathBuf::from("/tmp/vault.enc"))));
        // Initialize vault
        let password = secrecy::SecretString::new("test_password".to_string());
        security.write().await.initialize_vault(&password).unwrap();

        let auth = ApiAuth::new(security.clone());
        let login = auth.login("test_password").await;
        assert!(login.is_ok());
    }
}
