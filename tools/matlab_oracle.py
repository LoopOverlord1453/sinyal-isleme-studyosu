"""MATLAB kahini: kontrol / sistem tanımlama / MPC hesaplarının referans değerlerini MATLAB'in kendisinden alır.

Kullanım (test ya da betikten):

    from tools.matlab_oracle import fixture
    out = fixture("c2d_zoh", '''
        G = tf(1, [1 2 1]);
        Gd = c2d(G, 0.1, 'zoh');
        [n, d] = tfdata(Gd, 'v');
        out.num = n; out.den = d;
    ''')
    out["num"]  # → list

- Betik `out` adlı bir struct'ın alanlarını doldurur; sonuç JSON'a çevrilir (`jsonencode`). Karmaşık sayılar JSON'a
  giremez: `spt_c(x)` yardımcısı [gerçel, sanal] sütunlarına ayırır (ör. `out.p = spt_c(pole(G));`).
- Sonuç `tests/data/matlab/<ad>.json` dosyasına betiğin özetiyle (sha256) yazılır. Betik değişmediyse dosyadan okunur:
  testler MATLAB olmadan da çalışır. Betik değişmişse ve MATLAB yoksa `OracleUnavailable` (testte pytest.skip).
- MATLAB çağrıları kilit dosyasıyla sıraya konur: paralel çalışan testler / ajanlar aynı anda birden çok MATLAB
  oturumu açmaz (her oturum ~45 s açılış, 1–2 GB bellek).
- MATLAB yolu: SPT_MATLAB ortam değişkeni ya da C:\\Program Files\\MATLAB\\R20xx\\bin\\matlab.exe (en yenisi).

Komut satırı:  python tools/matlab_oracle.py --check        (MATLAB sürümü ve araç kutuları)
               python tools/matlab_oracle.py --run dosya.m  (betiği çalıştırıp JSON'u yazdırır)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "data" / "matlab"
LOCK_FILE = Path(tempfile.gettempdir()) / "spt_matlab_oracle.lock"
_HELPERS = r"""
function r = spt_c(x)
% Karmaşık diziyi JSON için [gerçel, sanal] sütunlarına ayırır (sütun vektörü olarak).
x = x(:);
r = [real(x), imag(x)];
end
"""


class OracleUnavailable(RuntimeError):
    """MATLAB yok ya da çalışmadı ve kayıtlı referans da yok."""


class OracleError(RuntimeError):
    """MATLAB betiği hata verdi (ayrıntı iletide)."""


def matlab_path() -> str | None:
    env = os.environ.get("SPT_MATLAB")
    if env and Path(env).exists():
        return env
    found = sorted(glob.glob(r"C:\Program Files\MATLAB\R20*\bin\matlab.exe"))
    return found[-1] if found else None


def _acquire(lock: Path, timeout: float):
    """Süreçler arası basit kilit (O_EXCL); sahibi ölmüş kilit 2 saat sonra bayat sayılır."""
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            return fd
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 7200:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout:
                raise OracleUnavailable(f"MATLAB kilidi {timeout:.0f} s içinde alınamadı: {lock}") from None
            time.sleep(1.0)


def run(script: str, timeout: float = 1800.0) -> dict:
    """MATLAB betiğini çalıştırır, `out` struct'ını sözlük olarak döndürür (önbelleksiz)."""
    exe = matlab_path()
    if exe is None:
        raise OracleUnavailable("MATLAB bulunamadı (SPT_MATLAB ile yol verilebilir)")
    fd = _acquire(LOCK_FILE, timeout)
    try:
        with tempfile.TemporaryDirectory(prefix="spt_oracle_") as tmp:
            wd = Path(tmp)
            (wd / "spt_c.m").write_text(_HELPERS.strip() + "\n", encoding="utf-8")
            out_path, err_path = wd / "out.json", wd / "err.txt"
            body = "\n".join("    " + line for line in script.strip("\n").splitlines())
            main = (f"function spt_oracle_main()\ntry\n    out = struct();\n{body}\n"
                    f"    fid = fopen('{out_path.as_posix()}', 'w', 'n', 'UTF-8');\n"
                    "    fwrite(fid, jsonencode(out, 'ConvertInfAndNaN', false), 'char');\n    fclose(fid);\n"
                    "catch err\n"
                    f"    fid = fopen('{err_path.as_posix()}', 'w', 'n', 'UTF-8');\n"
                    "    fwrite(fid, getReport(err, 'extended', 'hyperlinks', 'off'), 'char');\n    fclose(fid);\n"
                    "end\nend\n")
            (wd / "spt_oracle_main.m").write_text(main, encoding="utf-8")
            r = subprocess.run([exe, "-batch", "spt_oracle_main"], cwd=wd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
            if err_path.exists():
                raise OracleError(err_path.read_text(encoding="utf-8", errors="replace"))
            if not out_path.exists():
                raise OracleError(f"MATLAB çıktı üretmedi (kod {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        os.close(fd)
        LOCK_FILE.unlink(missing_ok=True)


def _sha(script: str) -> str:
    return hashlib.sha256(script.strip().encode("utf-8")).hexdigest()


def fixture(name: str, script: str, *, regenerate: bool = False) -> dict:
    """Kayıtlı MATLAB referansı (betik değişmediyse dosyadan, değiştiyse MATLAB'den yeniden üretilir)."""
    path = FIXTURE_DIR / f"{name}.json"
    sha = _sha(script)
    if path.exists() and not regenerate:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("script_sha") == sha:
            return data["out"]
    out = run(script)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"script_sha": sha, "matlab": "R2025b", "script": script.strip(), "out": out},
                               ensure_ascii=False, indent=1, allow_nan=True), encoding="utf-8")
    return out


def _main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--check" in argv:
        out = run("v = ver; out.version = version; out.toolboxes = {v.Name};"
                  "out.nan = NaN; out.inf = [Inf -Inf]; out.c = spt_c([1+2i; 3-4i]); out.m = [1 2; 3 4];")
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0
    if "--run" in argv:
        script = Path(argv[argv.index("--run") + 1]).read_text(encoding="utf-8")
        print(json.dumps(run(script), ensure_ascii=False, indent=1))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
