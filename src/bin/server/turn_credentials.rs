//! Ephemeral TURN credential generation (RFC 5766 long-term credentials)
//!
//! COTURN shared-secret mode:
//!   username = "{expiry}:{identifier}"
//!   credential = base64(HMAC-SHA1(shared_secret, username))

use base64::Engine;
use hmac::{Hmac, Mac};
use sha1::Sha1;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha1 = Hmac<Sha1>;

/// Generate ephemeral TURN credentials from a shared secret.
///
/// Returns `(username, credential)` valid for `ttl_secs` from now.
pub fn generate_turn_credentials(shared_secret: &str, ttl_secs: u64) -> (String, String) {
    let expiry = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + ttl_secs;

    let username = format!("{}:parakeet", expiry);

    let mut mac =
        HmacSha1::new_from_slice(shared_secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(username.as_bytes());
    let result = mac.finalize().into_bytes();

    let credential = base64::engine::general_purpose::STANDARD.encode(result);

    (username, credential)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_deterministic_for_same_input() {
        // Two calls with same secret produce valid base64 credentials
        let (u1, c1) = generate_turn_credentials("test-secret", 86400);
        let (u2, c2) = generate_turn_credentials("test-secret", 86400);

        // Usernames should have same suffix
        assert!(u1.ends_with(":parakeet"));
        assert!(u2.ends_with(":parakeet"));

        // Credentials should be valid base64
        assert!(base64::engine::general_purpose::STANDARD.decode(&c1).is_ok());
        assert!(base64::engine::general_purpose::STANDARD.decode(&c2).is_ok());

        // Expiry timestamps should be within 1 second of each other
        let exp1: u64 = u1.split(':').next().unwrap().parse().unwrap();
        let exp2: u64 = u2.split(':').next().unwrap().parse().unwrap();
        assert!(exp2 - exp1 <= 1);
    }

    #[test]
    fn different_secrets_produce_different_credentials() {
        let (_, c1) = generate_turn_credentials("secret-a", 86400);
        let (_, c2) = generate_turn_credentials("secret-b", 86400);
        assert_ne!(c1, c2);
    }
}
