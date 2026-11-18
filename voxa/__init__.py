"""Voxa application package."""

import os

APP_ID = "io.github.crhy.voxa"
APP_NAME = "Voxa"
APP_VERSION = "0.1.1"
# Optional build-channel tag (e.g. VOXA_BUILD_TAG=test) so a side test build
# is recognizable in the window title and About dialog. Empty by default, so
# production looks exactly as before.
BUILD_TAG = os.environ.get("VOXA_BUILD_TAG", "").strip()
APP_DISPLAY_NAME = f"{APP_NAME} [{BUILD_TAG}]" if BUILD_TAG else APP_NAME
APP_DISPLAY_VERSION = f"{APP_VERSION} ({BUILD_TAG})" if BUILD_TAG else APP_VERSION
