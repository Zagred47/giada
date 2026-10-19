"""Isolate official CLI OAuth/session/history stores by explicit account profile.

Credential loading and Google consent remain entirely in the official SDK.
This wrapper never reads, copies or logs credential contents.
"""
import os
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
    state._store=StateStore(str(folder/'sessions.json'))
    state._settings_store=SettingsStore(str(folder/'settings.json'))
    state._history=HistoryLogger(str(folder/'history'))

if __name__=='__main__':
    configure_profile()
    from colab_cli.cli import app
    app()
