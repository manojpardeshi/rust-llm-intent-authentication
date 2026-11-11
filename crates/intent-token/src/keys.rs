//! Ed25519 signing keys, on-disk key files and the public JWK set.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::SigningKey;
use ed25519_dalek::pkcs8::EncodePrivateKey as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::TokenError;

/// An Ed25519 key pair identified by a `kid`. The signing key is zeroised
/// on drop (ed25519-dalek's `zeroize` feature), as are the intermediate
/// seed, DER and key-file buffers.
pub struct KeyPair {
    kid: String,
    signing_key: SigningKey,
}

impl std::fmt::Debug for KeyPair {
    // Never print private key material.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyPair")
            .field("kid", &self.kid)
            .finish_non_exhaustive()
    }
}

impl KeyPair {
    /// Generates a new key from the OS RNG.
    pub fn generate() -> Result<Self, TokenError> {
        let mut seed = Zeroizing::new([0u8; 32]);
        getrandom::fill(seed.as_mut()).map_err(|e| TokenError::Key(e.to_string()))?;
        Ok(Self::from_seed_ref(&seed))
    }

    pub fn from_seed(seed: [u8; 32]) -> Self {
        let seed = Zeroizing::new(seed);
        Self::from_seed_ref(&seed)
    }

    fn from_seed_ref(seed: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(seed);
        let kid = thumbprint(&public_x(&signing_key));
        Self { kid, signing_key }
    }

    pub fn kid(&self) -> &str {
        &self.kid
    }

    /// Base64url public key (the JWK `x` member).
    pub fn public_x(&self) -> String {
        public_x(&self.signing_key)
    }

    pub fn public_jwk(&self) -> Jwk {
        Jwk {
            kty: "OKP".into(),
            crv: "Ed25519".into(),
            alg: "EdDSA".into(),
            use_: "sig".into(),
            kid: self.kid.clone(),
            x: self.public_x(),
        }
    }

    pub(crate) fn pkcs8_der(&self) -> Result<Zeroizing<Vec<u8>>, TokenError> {
        self.signing_key
            .to_pkcs8_der()
            .map(|doc| Zeroizing::new(doc.as_bytes().to_vec()))
            .map_err(|e| TokenError::Key(e.to_string()))
    }

    /// Writes the key as JSON, readable only by the owner on Unix.
    pub fn save(&self, path: &Path) -> Result<(), TokenError> {
        let file = KeyFile {
            kid: self.kid.clone(),
            alg: "EdDSA".into(),
            d: Zeroizing::new(
                URL_SAFE_NO_PAD.encode(Zeroizing::new(self.signing_key.to_bytes()).as_ref()),
            ),
            x: self.public_x(),
        };
        let json = Zeroizing::new(
            serde_json::to_vec_pretty(&file).map_err(|e| TokenError::Key(e.to_string()))?,
        );

        let mut opts = fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            opts.mode(0o600);
        }
        let mut f = opts
            .open(path)
            .map_err(|e| TokenError::Key(e.to_string()))?;
        f.write_all(&json)
            .map_err(|e| TokenError::Key(e.to_string()))?;
        // Make sure the key is on disk before reporting success.
        f.sync_all().map_err(|e| TokenError::Key(e.to_string()))
    }

    pub fn load(path: &Path) -> Result<Self, TokenError> {
        let bytes = Zeroizing::new(fs::read(path).map_err(|e| TokenError::Key(e.to_string()))?);
        let file: KeyFile =
            serde_json::from_slice(&bytes).map_err(|e| TokenError::Key(e.to_string()))?;
        let decoded = Zeroizing::new(
            URL_SAFE_NO_PAD
                .decode(file.d.as_bytes())
                .map_err(|_| TokenError::Key("invalid private key encoding".into()))?,
        );
        let seed: &[u8; 32] = decoded
            .as_slice()
            .try_into()
            .map_err(|_| TokenError::Key("invalid private key encoding".into()))?;
        let key = Self::from_seed_ref(seed);
        if key.kid != file.kid || key.public_x() != file.x {
            return Err(TokenError::Key(
                "key file kid/x do not match private key".into(),
            ));
        }
        Ok(key)
    }
}

#[derive(Serialize, Deserialize)]
struct KeyFile {
    kid: String,
    alg: String,
    d: Zeroizing<String>,
    x: String,
}

/// A public Ed25519 JWK (RFC 8037).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Jwk {
    pub kty: String,
    pub crv: String,
    pub alg: String,
    #[serde(rename = "use")]
    pub use_: String,
    pub kid: String,
    pub x: String,
}

/// A JWK set, as served at `/.well-known/jwks.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Jwks {
    pub keys: Vec<Jwk>,
}

fn public_x(key: &SigningKey) -> String {
    URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes())
}

/// RFC 7638 JWK thumbprint for an Ed25519 key.
fn thumbprint(x: &str) -> String {
    let canonical = format!(r#"{{"crv":"Ed25519","kty":"OKP","x":"{x}"}}"#);
    URL_SAFE_NO_PAD.encode(Sha256::digest(canonical.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("signing.key.json");
        let key = KeyPair::generate().unwrap();
        key.save(&path).unwrap();

        let loaded = KeyPair::load(&path).unwrap();
        assert_eq!(loaded.kid(), key.kid());
        assert_eq!(loaded.public_x(), key.public_x());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn save_refuses_to_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.json");
        KeyPair::generate().unwrap().save(&path).unwrap();
        assert!(KeyPair::generate().unwrap().save(&path).is_err());
    }

    #[test]
    fn debug_hides_private_key() {
        let key = KeyPair::from_seed([7; 32]);
        let dbg = format!("{key:?}");
        assert!(dbg.contains(key.kid()));
        assert!(!dbg.contains(&URL_SAFE_NO_PAD.encode([7u8; 32])));
    }
}
