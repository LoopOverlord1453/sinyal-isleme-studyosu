"""Sinyal Test Uygulaması giriş noktası.

    python -m spt.sender                                   arayüz (son ayarlarla)
    python -m spt.sender --host 192.168.1.20 --port 50505  hedefi ver
    python -m spt.sender --scenario "IMU 6 eksen 1 kHz" --start
    python -m spt.sender --headless --scenario Hepsi --duration 10   arayüzsüz (konsol)
    python -m spt.sender --list                            senaryoları listele

Stüdyo, Araçlar → Sinyal Test Uygulaması ile bunu `--studio` bayrağıyla, kendi dinlediği adres ve portla açar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt6")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")


def build_parser() -> argparse.ArgumentParser:
    from spt.protocol.format import UDP_PORT

    p = argparse.ArgumentParser(prog="python -m spt.sender",
                                description="Gerçek zamanlı SPTF sinyal test uygulaması (UDP gönderici).")
    p.add_argument("--host", help="hedef IP (Stüdyo'nun çalıştığı bilgisayar; varsayılan son ayar / 127.0.0.1)")
    p.add_argument("--port", type=int, help=f"hedef UDP portu (varsayılan son ayar / {UDP_PORT})")
    p.add_argument("--bind", help="çıkış için yerel IPv4 (varsayılan: işletim sistemi seçer)")
    p.add_argument("--scenario", help="hazır senaryo adı (bkz. --list)")
    p.add_argument("--config", type=Path, help="JSON ayar dosyası (arayüzdeki Kaydet ile oluşturulur)")
    p.add_argument("--start", action="store_true", help="arayüz açılınca gönderimi başlat")
    p.add_argument("--studio", action="store_true", help="Stüdyo'dan açıldı (hedef Stüdyo'nun dinlediği port)")
    p.add_argument("--headless", action="store_true", help="arayüzsüz: konsolda gönder (Ctrl+C ile durur)")
    p.add_argument("--duration", type=float, default=0.0, help="arayüzsüz modda süre, s (0 = durdurulana kadar)")
    p.add_argument("--list", action="store_true", help="senaryoları listele ve çık")
    return p


def _config(args, base):
    from spt.sender import signals as sg

    cfg = base
    if args.config:
        cfg = sg.config_from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    if args.scenario:
        cfg.streams = sg.scenario(args.scenario)
        cfg.scenario = args.scenario
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.bind is not None:
        cfg.bind = args.bind
    return cfg


def _headless(cfg, duration: float) -> int:
    from spt.sender.engine import Engine, _fine_timer

    last = [0.0]

    def show(ev):   # konsola saniyede bir satır
        if ev[0] != "stats" or ev[1]["t"] - last[0] < 1.0:
            return
        st = ev[1]
        last[0] = st["t"]
        rows = " · ".join(f"{r['name']} {r['pps']:.0f} p/s" + (" ⚠" if r["late"] else "") for r in st["streams"])
        print(f"{st['t']:6.1f} s · {st['bytes_per_s'] / 1e6:7.2f} MB/s · {rows}", flush=True)

    engine = Engine(cfg, show)
    print(f"Gönderiliyor → {cfg.host}:{cfg.port} · {len(engine.streams)} akış (Ctrl+C: dur)", flush=True)
    _fine_timer(True)
    try:
        engine.run(duration=duration)
    except KeyboardInterrupt:
        engine.flush()
    finally:
        engine.close()
        _fine_timer(False)
    return 0


def main(argv: list[str] | None = None) -> int:
    from spt.console import utf8_stdio

    utf8_stdio()
    args = build_parser().parse_args(argv)
    from spt.sender import signals as sg

    if args.list:
        for name in sg.scenario_names():
            print(name)
        return 0
    if args.scenario and args.scenario not in sg.scenario_names():
        print(f"Hata: bilinmeyen senaryo: {args.scenario} (--list)", file=sys.stderr)
        return 2
    if args.headless:
        return _headless(_config(args, sg.default_config()), args.duration)

    from PyQt6.QtWidgets import QApplication

    from spt import ORG_NAME
    from spt.gui.theme import ThemeManager
    from spt.sender.window import APP_TITLE, SenderWindow, load_config, sender_settings

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName(ORG_NAME)
    theme = ThemeManager(app)
    theme.apply()
    settings = sender_settings()
    cfg = _config(args, load_config(settings))
    target = (cfg.host, cfg.port) if args.studio else None
    window = SenderWindow(theme, cfg, settings, studio_target=target)
    window.show()
    if args.start:
        window.start()
    code = app.exec()
    time.sleep(0.05)
    return code


if __name__ == "__main__":
    sys.exit(main())
