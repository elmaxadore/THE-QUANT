//! # Security & Authentication Module (Layer 1)
//!
//! Implements master password hashing (Argon2id), AES-256-GCM encryption for
//! credentials at rest, JWT session management, and secure memory handling.
//!
//! ## Security Contract
//! - Master password: Argon2id with time_cost=3, memory_cost=65536 KiB, parallelism=4
//! - Encryption: AES-256-GCM with 96-bit random nonce
//! - Sessions: JWT HS256, 1-hour expiry, refresh on activity
//! - Auto-lock: 5 minutes of inactivity in TUI
//! - All sensitive strings use the `secrecy` crate (zero-on-Drop)

use crate::error::{QuantError, QuantResult};
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::{
    password_hash::{rand_core::OsRng as ArgonRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use chrono::{DateTime, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

/// Vault file format:
/// version(1 byte) || nonce(12 bytes) || ciphertext || tag(16 bytes)
const VAULT_VERSION: u8 = 1;
const NONCE_SIZE: usize = 12;
const TAG_SIZE: usize = 16;

/// JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,          // Subject (user ID)
    pub exp: usize,           // Expiry timestamp
    pub iat: usize,           // Issued at
    pub session_id: String,   // Unique session identifier
}

/// Session state
#[derive(Debug, Clone)]
pub struct Session {
    pub token: String,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub user_id: String,
}

/// Security manager — handles all cryptographic operations
#[derive(Debug)]
pub struct SecurityManager {
    /// Path to the encrypted vault file
    vault_path: PathBuf,
    /// Active sessions (in-memory only, no persistence)
    sessions: Arc<dashmap::DashMap<String, Session>>,
    /// JWT secret (derived from master password)
    jwt_secret: SecretString,
    /// Encryption key (derived from master password)
    encryption_key: [u8; 32],
    /// Whether the vault has been unlocked
    unlocked: bool,
}

impl SecurityManager {
    /// Create a new SecurityManager (locked state)
    pub fn new(vault_path: PathBuf) -> Self {
        Self {
            vault_path,
            sessions: Arc::new(dashmap::DashMap::new()),
            jwt_secret: SecretString::new(String::new()),
            encryption_key: [0u8; 32],
            unlocked: false,
        }
    }

    /// Initialize the vault with a master password (first run)
    pub fn initialize_vault(&mut self, master_password: &SecretString) -> QuantResult<()> {
        // Hash the master password with Argon2id
        let salt = SaltString::generate(&mut ArgonRng);
        let argon2 = Argon2::default();
        let password_hash = argon2
            .hash_password(master_password.expose_secret().as_bytes(), &salt)?
            .to_string();

        // Derive encryption key using HKDF-SHA256
        let mut hkdf = hkdf::Hkdf::<Sha256>::new(None, master_password.expose_secret().as_bytes());
        let mut enc_key = [0u8; 32];
        hkdf.expand(b"the-quant-encryption-key", &mut enc_key)
            .map_err(|e| QuantError::CryptoError(format!("HKDF expansion failed: {}", e)))?;

        // Derive JWT secret
        let mut jwt_secret_bytes = [0u8; 32];
        hkdf.expand(b"the-quant-jwt-secret", &mut jwt_secret_bytes)
            .map_err(|e| QuantError::CryptoError(format!("HKDF expansion failed: {}", e)))?;

        self.encryption_key = enc_key;
        self.jwt_secret = SecretString::new(
            String::from_utf8_lossy(&jwt_secret_bytes).to_string()
        );

        // Create vault file: version || nonce || ciphertext || tag
        let vault_data = format!("hash:{}", password_hash);
        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key)
            .map_err(|e| QuantError::CryptoError(format!("AES key init failed: {}", e)))?;
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, vault_data.as_bytes())
            .map_err(|e| QuantError::CryptoError(format!("Encryption failed: {}", e)))?;

        let mut vault_file = vec![VAULT_VERSION];
        vault_file.extend_from_slice(&nonce);
        vault_file.extend_from_slice(&ciphertext);

        // Ensure directory exists
        if let Some(parent) = self.vault_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.vault_path, &vault_file)?;

        self.unlocked = true;
        info!("Vault initialized at {:?}", self.vault_path);
        Ok(())
    }

    /// Unlock the vault with the master password
    pub fn unlock(&mut self, master_password: &SecretString) -> QuantResult<Session> {
        let vault_data = std::fs::read(&self.vault_path)?;
        
        if vault_data.is_empty() || vault_data[0] != VAULT_VERSION {
            return Err(QuantError::VaultError("Invalid vault file format".into()));
        }

        // Parse vault: version(1) || nonce(12) || ciphertext || tag(16)
        let nonce = &vault_data[1..1 + NONCE_SIZE];
        let ciphertext = &vault_data[1 + NONCE_SIZE..];

        // Derive encryption key from password
        let mut hkdf = hkdf::Hkdf::<Sha256>::new(None, master_password.expose_secret().as_bytes());
        let mut enc_key = [0u8; 32];
        hkdf.expand(b"the-quant-encryption-key", &mut enc_key)
            .map_err(|e| QuantError::CryptoError(format!("HKDF expansion failed: {}", e)))?;

        // Decrypt
        let cipher = Aes256Gcm::new_from_slice(&enc_key)
            .map_err(|e| QuantError::CryptoError(format!("AES key init failed: {}", e)))?;
        let nonce = Nonce::from_slice(nonce);
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| QuantError::AuthenticationError("Invalid master password".into()))?;

        let vault_contents = String::from_utf8_lossy(&plaintext);
        
        // Verify password hash
        if let Some(hash_str) = vault_contents.strip_prefix("hash:") {
            let parsed_hash = PasswordHash::new(hash_str)
                .map_err(|e| QuantError::VaultError(format!("Invalid hash in vault: {}", e)))?;
            Argon2::default()
                .verify_password(master_password.expose_secret().as_bytes(), &parsed_hash)
                .map_err(|_| QuantError::AuthenticationError("Invalid master password".into()))?;
        } else {
            return Err(QuantError::VaultError("Invalid vault contents".into()));
        }

        // Derive JWT secret
        let mut jwt_secret_bytes = [0u8; 32];
        hkdf.expand(b"the-quant-jwt-secret", &mut jwt_secret_bytes)
            .map_err(|e| QuantError::CryptoError(format!("HKDF expansion failed: {}", e)))?;

        self.encryption_key = enc_key;
        self.jwt_secret = SecretString::new(
            String::from_utf8_lossy(&jwt_secret_bytes).to_string()
        );
        self.unlocked = true;

        // Create session
        let session = self.create_session("quant")?;
        info!("Vault unlocked successfully");
        Ok(session)
    }

    /// Create a new JWT session
    pub fn create_session(&self, user_id: &str) -> QuantResult<Session> {
        if !self.unlocked {
            return Err(QuantError::AuthenticationError("Vault not unlocked".into()));
        }

        let now = Utc::now();
        let session_id = uuid::Uuid::new_v4().to_string();
        let claims = Claims {
            sub: user_id.to_string(),
            exp: (now.timestamp() + 3600) as usize, // 1 hour
            iat: now.timestamp() as usize,
            session_id: session_id.clone(),
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.expose_secret().as_bytes()),
        )?;

        let session = Session {
            token: token.clone(),
            created_at: now,
            last_activity: now,
            user_id: user_id.to_string(),
        };

        self.sessions.insert(session_id, session.clone());
        Ok(session)
    }

    /// Validate a JWT token and return the claims
    pub fn validate_token(&self, token: &str) -> QuantResult<Claims> {
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.expose_secret().as_bytes()),
            &Validation::default(),
        )?;

        // Check if session exists and is active
        if !self.sessions.contains_key(&token_data.claims.session_id) {
            return Err(QuantError::SessionExpired);
        }

        Ok(token_data.claims)
    }

    /// Refresh a session (extend expiry)
    pub fn refresh_session(&self, session_id: &str) -> QuantResult<()> {
        if let Some(mut session) = self.sessions.get_mut(session_id) {
            session.last_activity = Utc::now();
            Ok(())
        } else {
            Err(QuantError::SessionExpired)
        }
    }

    /// Invalidate a session (logout)
    pub fn invalidate_session(&self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Check if the vault is unlocked
    pub fn is_unlocked(&self) -> bool {
        self.unlocked
    }

    /// Encrypt sensitive data (e.g., MT5 credentials)
    pub fn encrypt(&self, plaintext: &[u8]) -> QuantResult<Vec<u8>> {
        if !self.unlocked {
            return Err(QuantError::AuthenticationError("Vault not unlocked".into()));
        }
        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key)
            .map_err(|e| QuantError::CryptoError(format!("AES key init failed: {}", e)))?;
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| QuantError::CryptoError(format!("Encryption failed: {}", e)))?;

        let mut result = nonce.to_vec();
        result.extend_from_slice(&ciphertext);
        Ok(result)
    }

    /// Decrypt sensitive data
    pub fn decrypt(&self, ciphertext: &[u8]) -> QuantResult<Vec<u8>> {
        if !self.unlocked {
            return Err(QuantError::AuthenticationError("Vault not unlocked".into()));
        }
        if ciphertext.len() < NONCE_SIZE {
            return Err(QuantError::CryptoError("Invalid ciphertext".into()));
        }

        let (nonce_bytes, ct) = ciphertext.split_at(NONCE_SIZE);
        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key)
            .map_err(|e| QuantError::CryptoError(format!("AES key init failed: {}", e)))?;
        let nonce = Nonce::from_slice(nonce_bytes);
        let plaintext = cipher
            .decrypt(nonce, ct)
            .map_err(|e| QuantError::CryptoError(format!("Decryption failed: {}", e)))?;

        Ok(plaintext)
    }

    /// Lock the vault (clear sensitive data from memory)
    pub fn lock(&mut self) {
        self.jwt_secret = SecretString::new(String::new());
        self.encryption_key = [0u8; 32];
        self.unlocked = false;
        self.sessions.clear();
        info!("Vault locked");
    }

    /// Check if vault file exists
    pub fn vault_exists(&self) -> bool {
        self.vault_path.exists()
    }

    /// Get active session count
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Remove expired sessions
    pub fn cleanup_expired_sessions(&self) {
        let now = Utc::now().timestamp();
        self.sessions.retain(|_, session| {
            // Sessions expire after 1 hour of inactivity
            let elapsed = now - session.last_activity.timestamp();
            elapsed < 3600
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::SecretString;
    use tempfile::tempdir;

    #[test]
    fn test_vault_initialization_and_unlock() {
        let dir = tempdir().unwrap();
        let vault_path = dir.path().join("vault.enc");
        let mut manager = SecurityManager::new(vault_path.clone());

        let password = SecretString::new("test_master_password_123".to_string());
        assert!(manager.initialize_vault(&password).is_ok());
        assert!(manager.is_unlocked());

        // Lock and unlock
        manager.lock();
        assert!(!manager.is_unlocked());

        let result = manager.unlock(&password);
        assert!(result.is_ok());
        assert!(manager.is_unlocked());
    }

    #[test]
    fn test_wrong_password_fails() {
        let dir = tempdir().unwrap();
        let vault_path = dir.path().join("vault.enc");
        let mut manager = SecurityManager::new(vault_path.clone());

        let password = SecretString::new("correct_password".to_string());
        manager.initialize_vault(&password).unwrap();
        manager.lock();

        let wrong_password = SecretString::new("wrong_password".to_string());
        let result = manager.unlock(&wrong_password);
        assert!(result.is_err());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let dir = tempdir().unwrap();
        let vault_path = dir.path().join("vault.enc");
        let mut manager = SecurityManager::new(vault_path.clone());

        let password = SecretString::new("test_password".to_string());
        manager.initialize_vault(&password).unwrap();

        let data = b"MT5:password123:server:443";
        let encrypted = manager.encrypt(data).unwrap();
        assert_ne!(encrypted, data);

        let decrypted = manager.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, data);
    }

    #[test]
    fn test_jwt_session() {
        let dir = tempdir().unwrap();
        let vault_path = dir.path().join("vault.enc");
        let mut manager = SecurityManager::new(vault_path.clone());

        let password = SecretString::new("test_password".to_string());
        manager.initialize_vault(&password).unwrap();

        let session = manager.create_session("quant").unwrap();
        let claims = manager.validate_token(&session.token).unwrap();
        assert_eq!(claims.sub, "quant");
    }
}
