"""Uygulama ayarları (QSettings).

`SPT_SETTINGS_FILE` ortam değişkeni verilirse ayarlar o INI dosyasında tutulur; testler ve otomatik denetimler
böylece kullanıcının kayıtlı yerleşimine ve ayarlarına dokunmadan çalışır.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QSettings

from spt import ORG_NAME

APP_KEY = "SignalProcessingToolbox"


def app_settings() -> QSettings:
    path = os.environ.get("SPT_SETTINGS_FILE")
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings(ORG_NAME, APP_KEY)
