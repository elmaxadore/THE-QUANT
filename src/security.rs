//! SECURITY — Layer 1 (v2.1 §4).
//!
//! Single master password (Argon2id), encrypted vault with AES-256-GCM.
//! The vault is the ONLY place secrets may live; it is kept OUT of git.
//! Credentials for MT5, GitHub PAT, DB are encrypted here.
//!
//! Compile with the `crypto` feature (default). A fallback keeps a lean build
//! compiling, but production MUST use `crypto`.

use crate::config::Config;
use std::path::PathBuf;

/// A secret value that zeroes itself on drop.
#[derive(Clone)]
pub struct Secret(Vec<u8>);

impl Secret {
    pub fn new(s: impl AsRef<[u8]>) -> Self {
        Secret(s.as_ref().to_vec())
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.clone()
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        for b in self.0.iter_mut() {
            *b = 0;
        }
    }
}

/// Vault binary format:
///   version(u8) || salt(16) || nonce(12) || ciphertext || tag(16)
const VAULT_VERSION: u8 = 1;
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;
#[cfg(feature = "crypto")]
pub mod crypto {
    use super::*;
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use argon2::{Algorithm, Argon2, Params, Version};
    use rand::RngCore;

    /// Derive a 256-bit encryption key from the master password.
    pub fn derive_key(password: &[u8], salt: &[u8]) -> [u8; 32] {
        let params = Params::new(64 * 1024, 3, 4, Some(32)).expect("valid argon2 params");
        let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        let mut key = [0u8; 32];
        argon
            .hash_password_into(password, salt, &mut key)
            .expect("argon2 hash");
        key
    }

    pub fn hash_password(password: &[u8], salt: &[u8]) -> [u8; 32] {
        derive_key(password, salt)
    }

    pub fn encrypt(
        key: &[u8; 32],
        nonce: &[u8; NONCE_LEN],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, String> {
        let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| format!("cipher: {e}"))?;
        let n = Nonce::from_slice(nonce);
        cipher.encrypt(n, plaintext).map_err(|e| format!("encrypt: {e}"))
    }

    pub fn decrypt(
        key: &[u8; 32],
        nonce: &[u8; NONCE_LEN],
        ct: &[u8],
    ) -> Result<Vec<u8>, String> {
        let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| format!("cipher: {e}"))?;
        let n = Nonce::from_slice(nonce);
        cipher.decrypt(n, ct).map_err(|e| format!("decrypt: {e}"))
    }

    pub fn random_bytes<const N: usize>() -> [u8; N] {
        let mut b = [0u8; N];
        rand::rngs::OsRng.fill_bytes(&mut b);
        b
    }
/// Master vault bound to the configured state directory.
pub struct Vault {
    pub path: PathBuf,
}

impl Vault {
    pub fn new(cfg: &Config) -> Self {
        Vault { path: cfg.state_root().join("vault.enc") }
    }

    /// Create a new vault from a master password: writes salt + verifier, then
    /// returns the derived key for `seal_file`/`open_file`.
    pub fn init(&self, master_password: &str) -> Result<[u8; 32], String> {
        #[cfg(not(feature = "crypto"))]
        {
            let _ = master_password;
            return Err("vault requires `crypto` feature (default)".into());
        }

        #[cfg(feature = "crypto")]
        {
            if self.path.exists() {
                return Err("vault already exists".into());
            }
            let salt = crypto::random_bytes::<SALT_LEN>();
            let verifier = crypto::hash_password(master_password.as_bytes(), &salt);
            let mut blob = Vec::with_capacity(1 + SALT_LEN + 32);
            blob.push(VAULT_VERSION);
            blob.extend_from_slice(&salt);
            blob.extend_from_slice(&verifier);
            let dir = self.path.parent().unwrap_or(std::path::Path::new("."));
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            std::fs::write(&self.path, blob).map_err(|e| e.to_string())?;
            Ok(crypto::derive_key(master_password.as_bytes(), &salt))
        }
    }

    /// Load the vault and derive the encryption key from the master password.
    pub fn open(&self, master_password: &str) -> Result<[u8; 32], String> {
        #[cfg(not(feature = "crypto"))]
        {
            let _ = master_password;
            return Err("vault requires `crypto` feature (default)".into());
        }

        #[cfg(feature = "crypto")]
        {
            let blob = std::fs::read(&self.path).map_err(|e| format!("open vault: {e}"))?;
            if blob.len() < 1 + SALT_LEN + 32 || blob[0] != VAULT_VERSION {
                return Err("bad vault file".into());
            }
            let salt = &blob[1..1 + SALT_LEN];
            let expected = &blob[1 + SALT_LEN..1 + SALT_LEN + 32];
            let verifier = crypto::hash_password(master_password.as_bytes(), salt);
            use subtle::ConstantTimeEq;
            if verifier.ct_eq(expected).unwrap_u8() != 1 {
                return Err("wrong master password".into());
            }
            Ok(crypto::derive_key(master_password.as_bytes(), salt))
        }
    }

    /// Encrypt and store a named secret as a separate file in the vault dir.
    pub fn seal_file(&self, key: &[u8; 32], name: &str, plaintext: &[u8]) -> Result<(), String> {
        #[cfg(not(feature = "crypto"))]
        {
            let _ = (key, name, plaintext);
            return Err("sealing requires `crypto` feature".into());
        }

        #[cfg(feature = "crypto")]
        {
            let nonce = crypto::random_bytes::<NONCE_LEN>();
            let ct = crypto::encrypt(key, &nonce, plaintext)?;
            let mut blob = Vec::with_capacity(1 + NONCE_LEN + ct.len());
            blob.push(VAULT_VERSION);
            blob.extend_from_slice(&nonce);
            blob.extend_from_slice(&ct);
            let dir = self.path.parent().unwrap_or(std::path::Path::new("."));
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            std::fs::write(dir.join(format!("{name}.enc")), blob).map_err(|e| e.to_string())
        }
    }

    /// Open and decrypt a stored secret file.
    pub fn open_file(&self, key: &[u8; 32], name: &str) -> Result<Vec<u8>, String> {
        #[cfg(not(feature = "crypto"))]
        {
            let _ = (key, name);
            return Err("opening secrets requires `crypto` feature".into());
        }

        #[cfg(feature = "crypto")]
        {
            let dir = self.path.parent().unwrap_or(std::path::Path::new("."));
            let blob = std::fs::read(dir.join(format!("{name}.enc")))
                .map_err(|e| format!("read {name}.enc: {e}"))?;
            if blob.len() < 1 + NONCE_LEN || blob[0] != VAULT_VERSION {
                return Err("bad secret blob".into());
            }
            let nonce: [u8; NONCE_LEN] = blob[1..1 + NONCE_LEN].try_into().unwrap();
            crypto::decrypt(key, &nonce, &blob[1 + NONCE_LEN..])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "crypto")]
    fn vault_roundtrip() {
        let dir = std::env::temp_dir().join("tq_vault_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut cfg = Config::load().unwrap();
        cfg.system.state_dir = dir.to_str().unwrap().to_string();
        let v = Vault::new(&cfg);
        let key = v.init("hunter2").unwrap();
        v.seal_file(&key, "mt5_credentials", b"login=123456;password=secret").unwrap();
        let got = v.open_file(&key, "mt5_credentials").unwrap();
        assert_eq!(got, b"login=123456;password=secret");
        assert!(v.open("not-the-password").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
}