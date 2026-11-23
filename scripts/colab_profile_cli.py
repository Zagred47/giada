"""Isolate official CLI OAuth/session/history stores by explicit account profile.

Credential loading and Google consent remain entirely in the official SDK.
This wrapper never reads, copies or logs credential contents.
"""
import os
import sys
from pathlib import Path

def configure_profile():
    profile=os.environ.get('GIADA_COLAB_PROFILE','primary')
    if profile not in ('primary','reserve'):raise ValueError('Unknown Colab account profile')
    if profile=='primary':return
    from colab_cli import auth
    from colab_cli.common import state
    from colab_cli.state import StateStore,SettingsStore
    from colab_cli.history import HistoryLogger
    os.umask(0o077)
    folder=Path.home()/'.config/colab-cli-reserve';folder.mkdir(parents=True,mode=0o700,exist_ok=True)
    auth.TOKEN_CONFIG_PATH=str(folder/'token.json')
    # CLI0.8 renamed the legacy-store hook and introduced keyring namespaces.
    # Keep the reserve account isolated; never fall through to primary auth.
    if hasattr(auth,'LEGACY_TOKEN_PATH'):auth.LEGACY_TOKEN_PATH=str(folder/'token.json')
    if hasattr(auth,'KEYRING_SERVICE_NAME'):auth.KEYRING_SERVICE_NAME='google-colab-cli-reserve'
    state._store=StateStore(str(folder/'sessions.json'))
    state._settings_store=SettingsStore(str(folder/'settings.json'))
    state._history=HistoryLogger(str(folder/'history'))

if __name__=='__main__':
    configure_profile()
    # The official SSH proxy starts a fresh CLI process. Pass its documented
    # --config option so that child resolves the reserve session as well.
    if os.environ.get('GIADA_COLAB_PROFILE','primary')=='reserve' and '--config' not in sys.argv:
        sys.argv[1:1]=['--config',str(Path.home()/'.config/colab-cli-reserve/sessions.json')]
    from colab_cli.cli import app
    app()
