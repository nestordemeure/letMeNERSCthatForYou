import json
from pathlib import Path
from typing import Optional, Union
from authlib.jose import JsonWebKey
from authlib.oauth2.rfc7523 import PrivateKeyJWT
from authlib.integrations.httpx_client.oauth2_client import OAuth2Client

SFAPI_TOKEN_URL = "https://oidc.nersc.gov/c2id/token"
SFAPI_BASE_URL = "https://api.nersc.gov/api/v1.2"

class SFAPIOAuthClient:
    """Adapted from SFAPI's Client code"""
    def __init__(self, client_id: Optional[str] = None, secret: Optional[str] = None, token_url: Optional[str] = SFAPI_TOKEN_URL, api_base_url: Optional[str] = SFAPI_BASE_URL, key: Optional[Union[str, Path]] = None):
        if any(arg is None for arg in [client_id, secret]):
            self._read_client_secret_from_file(key)
        else:
            self.client_id = client_id
            self.secret = secret
        is_dev_base_url = ('dev' in api_base_url)
        is_dev_token_url = ('dev' in token_url)
        if is_dev_base_url and not is_dev_token_url:
            print(f"WARNING: you might be using a dev base API url and a normal token url! (api_base_url:{api_base_url} token_url:{token_url})")
        elif is_dev_token_url and not is_dev_base_url:
            print(f"WARNING: you might be using a dev token url and a normal API url! (api_base_url:{api_base_url} token_url:{token_url})")
        self.token_url = token_url
        self.api_base_url = api_base_url            
        self.key = key if key else None
        self.oauth2_session = None

    def _read_client_secret_from_file(self, name):
        if name is not None and Path(name).exists():
            # If the user gives a full path, then use it
            key_path = Path(name)
        else:
            # If not let's search in ~/.superfacility for the name or any key
            nickname = "" if name is None else name
            keys = Path().home() / ".superfacility"
            key_paths = list(keys.glob(f"{nickname}*"))
            key_path = None
            if len(key_paths) >= 1:
                key_path = Path(key_paths[0])
                if len(key_paths) > 1:
                    print(f"WARNING: {keys} folder contains more than one key, picked {key_path}")

        # We have no credentials
        if key_path is None or key_path.is_dir():
            return

        # Check that key is read only in case it's not
        # 0o100600 means chmod 600
        if key_path.stat().st_mode != 0o100600:
            raise RuntimeError(f"Incorrect permissions on the key. To fix run: chmod 600 {key_path}")

        with Path(key_path).open() as secret:
            if key_path.suffix == ".json":
                # Json file in the format {"client_id": "", "secret": ""}
                json_web_key = json.loads(secret.read())
                self.secret = JsonWebKey.import_key(json_web_key["secret"])
                self.client_id = json_web_key["client_id"]
            else:
                self.secret = secret.read()
                # Read in client_id from first line of file
                self.client_id = self.secret.split("\n")[0]

        # Get just client_id in case of spaces
        self.client_id = self.client_id.strip(" ")

        # Validate we got a correct looking client_id
        if len(self.client_id) != 13:
            raise RuntimeError(f"client_id not found in file {key_path}")

    def _create_oauth2_session(self):
        self.oauth2_session = OAuth2Client(
            client_id=self.client_id,
            client_secret=self.secret,
            token_endpoint_auth_method=PrivateKeyJWT(self.token_url),
            grant_type="client_credentials",
            token_endpoint=self.token_url,
            timeout=10.0,
        )
        self.oauth2_session.fetch_token()

    def get_oauth2_session(self):
        if not self.oauth2_session:
            self._create_oauth2_session()
        else:
            self.oauth2_session.ensure_active_token(self.oauth2_session.token)
        return self.oauth2_session

    def get_authorization_header(self):
        oauth2_session = self.get_oauth2_session()
        return {"Authorization": f"Bearer {oauth2_session.token['access_token']}"}
