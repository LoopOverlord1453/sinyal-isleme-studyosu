"""UDP alım kapasitesi ölçümü: simülatör(ler) → alıcı süreci, localhost üzerinde.

Kullanım:
    .venv\\Scripts\\python tools\\udp_benchmark.py              (varsayılan senaryolar)
    .venv\\Scripts\\python tools\\udp_benchmark.py --seconds 10

Her senaryoda bir veya birkaç simülatör hedef hızda gerçek zamanlı gönderir. Alım hızı, alıcının
istatistiklerinden gerçek geçen süreye göre hesaplanır; kayıp, gönderilen ve alınan örnek sayılarının
farkıdır (alıcının kendi kayıp sayacı da ayrıca gösterilir).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spt.console import utf8_stdio  # noqa: E402
from spt.sources.receiver import ReceiverConfig, run_udp_receiver  # noqa: E402

def _free_port() -> int:
    """Boş bir UDP portu (sabit port başka bir uygulamayla çakışabilir)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = _free_port()
DRAIN_S = 10.0  # göndericiler bitince alıcının birikmiş datagramları işlemesi için en fazla bekleme
SUMMARY = re.compile(r"BİTTİ · (\d+) çerçeve · (\d+) örnek · ([\d.]+) MB · ortalama ([\d.]+) MB/s")
SCENARIOS = [
    # (ad, gönderici sayısı, gönderici başına fs, kanal, tip, datagram)
    ("8 MB/s · 1472 B", 1, 1_000_000, 4, "int16", 1472),
    ("115 MB/s · 1472 B · 3 gönderici (gigabit, standart MTU)", 3, 4_791_667, 4, "int16", 1472),
    ("115 MB/s · 8972 B (gigabit, jumbo)", 1, 14_375_000, 4, "int16", 8972),
    ("115 MB/s · 65507 B (gigabit, büyük datagram)", 1, 14_375_000, 4, "int16", 65507),
]


def run_case(name: str, senders: int, fs: float, channels: int, type_name: str, datagram: int,
             seconds: float) -> None:
    ctx = mp.get_context("spawn")
    commands, events = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=run_udp_receiver,
                       args=(ReceiverConfig("127.0.0.1", PORT, socket_buffer_mb=256), commands, events))
    proc.start()
    while True:
        ev = events.get(timeout=30)
        if ev[0] == "state":
            break
        if ev[0] == "error":
            raise RuntimeError(ev[1])

    sims = [subprocess.Popen(
        [sys.executable, "-u", "-m", "spt.sources.simulator", "--port", str(PORT), "--signal", "sine",
         "--fs", str(fs), "--channels", str(channels), "--type", type_name, "--datagram", str(datagram),
         "--duration", str(seconds), "--stream-id", str(i + 1)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for i in range(senders)]

    samples: dict[str, int] = {}
    lost: dict[str, int] = {}
    curve: list[tuple[float, int]] = []  # (zaman, toplam bayt)
    finished_at = None
    deadline = time.time() + seconds + 60
    while time.time() < deadline:
        if finished_at is None and all(s.poll() is not None for s in sims):
            finished_at = time.time()
        if finished_at is not None and time.time() - finished_at > 1.0:
            # Alıcı hâlâ birikmiş datagramları işliyorsa bekle (aksi halde işlenmemiş birikim "kayıp" görünür)
            if len(curve) < 2 or curve[-1][1] == curve[-2][1] or time.time() - finished_at > DRAIN_S:
                break
        try:
            ev = events.get(timeout=0.2)
        except queue.Empty:
            continue
        if ev[0] == "stats":
            for st in ev[2]:
                samples[st.key], lost[st.key] = st.samples, st.lost_samples
            curve.append((time.monotonic(), sum(st.bytes for st in ev[2])))
    commands.put(("stop",))
    proc.join(5)

    sent = 0
    sender_rates = []
    for s in sims:
        match = SUMMARY.search(s.stdout.read())
        if not match:
            raise RuntimeError("simülatör çıktısı çözülemedi")
        sent += int(match.group(2))
        sender_rates.append(float(match.group(4)))

    active = [(t, b) for t, b in curve if b > 0]
    moving = [p for i, p in enumerate(active) if i == 0 or p[1] != active[i - 1][1]]
    rx_rate = (moving[-1][1] - moving[0][1]) / (moving[-1][0] - moving[0][0]) / 1e6 if len(moving) > 1 else 0.0
    received = sum(samples.values())
    missing = sent - received
    print(f"{name}\n"
          f"  gönderim: {' + '.join(f'{r:.1f}' for r in sender_rates)} MB/s · "
          f"alım (gerçek süreye göre): {rx_rate:.1f} MB/s\n"
          f"  kayıp: {missing} / {sent} örnek ({100 * missing / max(sent, 1):.4f} %) · "
          f"alıcının saydığı kayıp: {sum(lost.values())}\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    utf8_stdio()
    print(f"Python {sys.version.split()[0]} · her senaryo {args.seconds:g} s\n", flush=True)
    for scenario in SCENARIOS:
        run_case(*scenario, args.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
