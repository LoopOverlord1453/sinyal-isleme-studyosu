"""Arka plan hesap yürütücüsü: işler kalıcı bir yardımcı süreçte çalışır, arayüz donmaz, iş iptal edilebilir.

- İlk iş geldiğinde `spawn` ile bir yardımcı süreç açılır (numpy / scipy / sysid içe aktarımı bir kez ödenir) ve
  sonraki işler aynı süreçte çalışır. İletişim çift yönlü borudur; arayüz zamanlayıcıyla yoklar.
- İptal: süreç sonlandırılır (hesap gerçekten durur), sonraki işte yenisi açılır.
- İş argümanları seçilemiyorsa (pickle) iş arka plan iş parçacığında çalışır; iptal sonucu yok sayar (hesap
  arka planda biter) — bu yedek yol yalnız seçilemeyen model nesneleri için vardır.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import queue
import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from spt.gui.control.sysid_jobs import call_job, error_text, worker_main

log = logging.getLogger("spt.gui")


class JobRunner(QObject):
    """Tek seferde bir iş. Sinyaller: `progress(metin)`, `finished(tür, sonuç)`, `failed(tür, ileti)`."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(self, parent: QObject | None = None, poll_ms: int = 30):
        super().__init__(parent)
        self._proc = None
        self._conn = None
        self._job: tuple[int, str] | None = None     # (iş no, tür)
        self._next_id = 0
        self._thread_q: queue.Queue | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self) -> bool:
        return self._job is not None

    @property
    def kind(self) -> str:
        return self._job[1] if self._job else ""

    @property
    def process(self):
        """Yardımcı süreç (yoksa None) — testler ve tanılama için."""
        return self._proc

    # ---------------------------------------------------------------- başlat
    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._drop_worker()
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=worker_main, args=(child,), name="spt-sysid", daemon=True)
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent

    def warm(self) -> None:
        """Yardımcı süreci boştayken önceden açar (ilk işte numpy / scipy içe aktarımı beklenmez)."""
        if not self.busy:
            try:
                self._ensure_worker()
            except OSError as exc:  # pragma: no cover - süreç açılamadı: ilk işte yeniden denenir
                log.warning("Sistem tanımlama yardımcı süreci açılamadı: %s", exc)

    def start(self, kind: str, func, *args) -> None:
        if self.busy:
            raise RuntimeError("Önceki iş sürüyor")
        self._next_id += 1
        job_id = self._next_id
        try:
            payload = pickle.dumps((job_id, func, args), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - seçilemeyen nesne: iş parçacığına düş
            payload = None
        self._job = (job_id, kind)
        if payload is None:
            self._thread_q = q = queue.Queue()

            def run():
                try:
                    res = call_job(func, args, lambda t: q.put(("progress", job_id, t)))
                    q.put(("ok", job_id, res))
                except Exception as exc:  # noqa: BLE001
                    q.put(("error", job_id, error_text(exc), ""))

            threading.Thread(target=run, name="spt-sysid-thread", daemon=True).start()
        else:
            self._ensure_worker()
            self._conn.send_bytes(payload)
        self._timer.start()

    # ---------------------------------------------------------------- yokla
    def _messages(self):
        if self._thread_q is not None:
            while True:
                try:
                    yield self._thread_q.get_nowait()
                except queue.Empty:
                    return
        conn = self._conn
        if conn is None:
            return
        while True:
            try:
                if not conn.poll():
                    return
                yield conn.recv()
            except (EOFError, OSError):
                yield ("dead",)
                return

    def _poll(self) -> None:
        if self._job is None:
            self._timer.stop()
            return
        job_id, kind = self._job
        for msg in self._messages():
            if msg[0] == "dead":
                self._finish()
                self._drop_worker()
                self.failed.emit(kind, "Hesap süreci beklenmedik biçimde sonlandı")
                return
            if len(msg) < 2 or msg[1] != job_id:
                continue  # iptal edilmiş eski işin iletisi
            if msg[0] == "progress":
                self.progress.emit(msg[2])
            elif msg[0] == "ok":
                self._finish()
                self.finished.emit(kind, msg[2])
                return
            elif msg[0] == "error":
                self._finish()
                if len(msg) > 3 and msg[3]:
                    log.warning("Sistem tanımlama hesabı hata verdi: %s\n%s", msg[2], msg[3])
                self.failed.emit(kind, msg[2])
                return
        if self._thread_q is None and self._proc is not None and not self._proc.is_alive():
            self._finish()
            self._drop_worker()
            self.failed.emit(kind, "Hesap süreci beklenmedik biçimde sonlandı")

    def _finish(self) -> None:
        self._job = None
        self._thread_q = None
        self._timer.stop()

    # ---------------------------------------------------------------- iptal / kapat
    def cancel(self) -> bool:
        """Süren işi durdurur (süreç sonlandırılır). Döner: iş vardı mı."""
        if self._job is None:
            return False
        threaded = self._thread_q is not None
        self._finish()
        if not threaded:
            self._drop_worker()
        return True

    def _drop_worker(self) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:  # pragma: no cover
                pass
        if proc is not None:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=2.0)

    def shutdown(self) -> None:
        self.cancel()
        self._drop_worker()
