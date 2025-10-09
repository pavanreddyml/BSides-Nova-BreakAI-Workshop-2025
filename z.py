import base64
import hmac
import hashlib
import json

# This MUST be the same key used in your login.html's JavaScript
SECRET_KEY = 'this-is-a-shared-secret-for-the-demo'

def b64_url_decode(data):
    padding = '=' * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)

def verify_token(token):
    try:
        # 1. Split the token into its three parts.
        header_b64, payload_b64, signature_b64 = token.split('.')
        
        # 2. Recreate the data that was originally signed.
        signed_data = f"{header_b64}.{payload_b64}".encode('utf-8')
        
        # 3. Decode the signature provided by the client.
        decoded_signature = b64_url_decode(signature_b64)
        
        # 4. Generate what the signature *should* be using the secret key.
        expected_signature = hmac.new(
            SECRET_KEY.encode('utf-8'),
            signed_data,
            hashlib.sha256
        ).digest()
        
        # 5. Securely compare the two signatures.
        if hmac.compare_digest(decoded_signature, expected_signature):
            # If they match, decode the payload and return the username.
            payload = json.loads(b64_url_decode(payload_b64))
            return payload.get('username')
            
    except Exception:
        return None
    
    return None

# --- Example Usage ---
# good_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFsaWNlIn0.Zf-8_G3t6k_2Y-a_... (a valid token from login.html)"
# bad_token = "some.tampered.token"

# username = verify_token(good_token)
# if username:
#     print(f"Token is valid for user: {username}")
# else:
#     print("Token is invalid.")