"""Kontrol paketinin ortak hata ve uyarı türleri."""
from __future__ import annotations


class ControlError(ValueError):
    """Kullanıcıya doğrudan gösterilebilen Türkçe hata (geçersiz model, uyumsuz Ts, çözümsüz tasarım …).

    İleti arayüzdeki adları kullanır; iç API adı içermez."""


class ControlWarning(UserWarning):
    """Hesap yapıldı ama sonuç dikkat istiyor (kötü koşullu dönüşüm, Padé yaklaşımı, kararsız kapalı çevrim …)."""
