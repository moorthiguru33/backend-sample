"""
get_refresh_token.py
====================
Run this script ONCE on your LOCAL machine (not GitHub Actions).
It opens a browser, you login with your Google account, and prints
the 3 secrets you need to add to GitHub.

Install first:
  pip install google-auth-oauthlib

How to get CLIENT_ID and CLIENT_SECRET:
  1. Go to https://console.cloud.google.com
  2. Select your project (or create one)
  3. APIs & Services → Enable "Google Drive API"
  4. APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client IDs
  5. Application type: "Desktop app"
  6. Download or copy Client ID and Client Secret
  7. Paste them below and run this script
"""

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]

# ── PASTE YOUR VALUES HERE ─────────────────────────────────────────────────
CLIENT_ID     = "YOUR_CLIENT_ID_HERE.apps.googleusercontent.com"
CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
# ──────────────────────────────────────────────────────────────────────────

client_config = {
    "installed": {
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
        "token_uri":     "https://oauth2.googleapis.com/token",
        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
    }
}

print("🌐 Opening browser for Google login...")
print("   Login with the Google account that OWNS the Drive folders.\n")

flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

print("\n" + "=" * 60)
print("✅  COPY THESE 3 VALUES TO GITHUB SECRETS")
print("    Settings → Secrets and variables → Actions → New secret")
print("=" * 60)
print(f"\nGOOGLE_CLIENT_ID      =  {creds.client_id}")
print(f"GOOGLE_CLIENT_SECRET  =  {creds.client_secret}")
print(f"GOOGLE_REFRESH_TOKEN  =  {creds.refresh_token}")
print("\n" + "=" * 60)
print("⚠️  Never commit these values to Git.")
print("⚠️  Refresh token does NOT expire (unless revoked).")
