"""Yerel ağ arayüzleri: ad, tür, durum, bağlantı hızı, MTU, IPv4 adres / ağ maskesi.

Windows'ta iphlpapi `GetAdaptersAddresses` (ctypes) kullanılır: bağlantı hızı (TransmitLinkSpeed / ReceiveLinkSpeed)
yalnız buradan okunabilir. Linux'ta /sys/class/net, diğer sistemlerde yalnız adres bilgisi (QtNetwork varsa).
Bilinmeyen alanlar None'dır; hiçbir hata dışarı sızmaz (bilgi amaçlı).
Bu modül Qt'yi yalnız son çare olarak ve fonksiyon içinde içe aktarır.
"""
from __future__ import annotations

import ipaddress
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

ETHERNET, WIFI, LOOPBACK, OTHER = "Ethernet", "Wi-Fi", "Geri döngü", "Diğer"
# Sürücü adında bunlar geçen arayüzler sanal sayılır (Hyper-V, VirtualBox, VPN, Wi-Fi Direct…)
_VIRTUAL_HINTS = ("virtual", "hyper-v", "vmware", "vpn", "tap-", "wan miniport", "bluetooth", "loopback", "wsl")


@dataclass(frozen=True)
class NetInterface:
    name: str                       # kullanıcının gördüğü ad (Windows: "Ethernet 2")
    description: str                # sürücü / kart adı
    kind: str                       # ETHERNET | WIFI | LOOPBACK | OTHER
    up: bool
    speed_bps: int | None           # alım yönü bağlantı hızı (bit/s); bilinmiyorsa None
    mtu: int | None
    addresses: tuple[tuple[str, int], ...]   # (IPv4, önek uzunluğu)

    @property
    def virtual(self) -> bool:
        text = f"{self.name} {self.description}".lower()
        return any(h in text for h in _VIRTUAL_HINTS)

    @property
    def speed_text(self) -> str:
        return format_link_speed(self.speed_bps)

    def address_text(self) -> str:
        return ", ".join(f"{ip}/{plen}" for ip, plen in self.addresses) or "—"

    def contains(self, ip: str) -> bool:
        """`ip` bu arayüzün doğrudan bağlı alt ağlarından birinde mi?"""
        try:
            target = ipaddress.IPv4Address(ip)
        except ValueError:
            return False
        for addr, plen in self.addresses:
            try:
                if target in ipaddress.IPv4Network(f"{addr}/{plen}", strict=False):
                    return True
            except ValueError:
                continue
        return False


def format_link_speed(bps: int | None) -> str:
    if not bps:
        return "bilinmiyor"
    if bps >= 1_000_000_000:
        return f"{bps / 1e9:g} Gbit/s"
    if bps >= 1_000_000:
        return f"{bps / 1e6:g} Mbit/s"
    return f"{bps / 1e3:g} kbit/s"


def interfaces() -> list[NetInterface]:
    """Sistemdeki IPv4 arayüzleri (hata durumunda boş liste)."""
    try:
        if sys.platform == "win32":
            return _windows_interfaces()
        if sys.platform.startswith("linux"):
            return _linux_interfaces()
        return _qt_interfaces()
    except Exception:  # noqa: BLE001 — bilgi amaçlı; arayüzü asla bozmamalı
        return []


def local_ip_towards(remote: str) -> str | None:
    """`remote` adresine giden yolda kullanılacak yerel IPv4 (yönlendirme tablosuna sorar; paket gönderilmez)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((remote, 9))
            return s.getsockname()[0]
    except OSError:
        return None


def interface_for(remote: str | None, items: list[NetInterface] | None = None) -> NetInterface | None:
    """Göndericinin (`remote`) verisinin geldiği yerel arayüz: önce aynı alt ağ, sonra yönlendirmenin seçtiği
    yerel adres. `remote` yoksa: çalışan, geri döngü olmayan arayüzlerden en hızlısı."""
    items = interfaces() if items is None else items
    if remote:
        if remote.startswith("127."):
            return next((i for i in items if i.kind == LOOPBACK), None)
        for item in items:
            if item.up and item.contains(remote):
                return item
        local = local_ip_towards(remote)
        if local:
            for item in items:
                if any(ip == local for ip, _ in item.addresses):
                    return item
        return None
    active = [i for i in items if i.up and i.kind != LOOPBACK and i.addresses]
    routable = [i for i in active if any(not ip.startswith("169.254.") for ip, _ in i.addresses)]
    return max(routable or active, key=lambda i: (not i.virtual, i.kind == ETHERNET, i.speed_bps or 0), default=None)


# ---------------------------------------------------------------------------------------------------- Windows
def _windows_interfaces() -> list[NetInterface]:
    import ctypes
    from ctypes import POINTER, c_char_p, c_int, c_ubyte, c_uint64, c_ulong, c_ushort, c_void_p, c_wchar_p

    class SOCKADDR_IN(ctypes.Structure):  # noqa: N801 — Windows adı
        _fields_ = [("family", c_ushort), ("port", c_ushort), ("addr", c_ubyte * 4), ("zero", c_ubyte * 8)]

    class SOCKET_ADDRESS(ctypes.Structure):  # noqa: N801
        _fields_ = [("lpSockaddr", c_void_p), ("iSockaddrLength", c_int)]

    class UNICAST(ctypes.Structure):
        pass

    UNICAST._fields_ = [("Length", c_ulong), ("Flags", c_ulong), ("Next", POINTER(UNICAST)),
                        ("Address", SOCKET_ADDRESS), ("PrefixOrigin", c_int), ("SuffixOrigin", c_int),
                        ("DadState", c_int), ("ValidLifetime", c_ulong), ("PreferredLifetime", c_ulong),
                        ("LeaseLifetime", c_ulong), ("OnLinkPrefixLength", c_ubyte)]

    class ADAPTER(ctypes.Structure):  # IP_ADAPTER_ADDRESSES_LH, ReceiveLinkSpeed'e kadar (gerisi okunmaz)
        pass

    ADAPTER._fields_ = [("Length", c_ulong), ("IfIndex", c_ulong), ("Next", POINTER(ADAPTER)),
                        ("AdapterName", c_char_p), ("FirstUnicastAddress", POINTER(UNICAST)),
                        ("FirstAnycastAddress", c_void_p), ("FirstMulticastAddress", c_void_p),
                        ("FirstDnsServerAddress", c_void_p), ("DnsSuffix", c_wchar_p), ("Description", c_wchar_p),
                        ("FriendlyName", c_wchar_p), ("PhysicalAddress", c_ubyte * 8),
                        ("PhysicalAddressLength", c_ulong), ("Flags", c_ulong), ("Mtu", c_ulong),
                        ("IfType", c_ulong), ("OperStatus", c_int), ("Ipv6IfIndex", c_ulong),
                        ("ZoneIndices", c_ulong * 16), ("FirstPrefix", c_void_p),
                        ("TransmitLinkSpeed", c_uint64), ("ReceiveLinkSpeed", c_uint64)]

    api = ctypes.WinDLL("iphlpapi")
    flags = 0x2 | 0x4 | 0x8          # GAA_FLAG_SKIP_ANYCAST | SKIP_MULTICAST | SKIP_DNS_SERVER
    size = c_ulong(16 * 1024)
    for _ in range(4):
        buf = ctypes.create_string_buffer(size.value)
        ret = api.GetAdaptersAddresses(socket.AF_INET, flags, None, buf, ctypes.byref(size))
        if ret == 0:
            break
        if ret != 111:                   # ERROR_BUFFER_OVERFLOW dışında: başarısız
            return []
    else:
        return []
    out = []
    node = ctypes.cast(buf, POINTER(ADAPTER))
    while node:
        a = node.contents
        addrs = []
        uni = a.FirstUnicastAddress
        while uni:
            u = uni.contents
            if u.Address.lpSockaddr and u.Address.iSockaddrLength >= ctypes.sizeof(SOCKADDR_IN):
                sa = ctypes.cast(u.Address.lpSockaddr, POINTER(SOCKADDR_IN)).contents
                if sa.family == socket.AF_INET:
                    addrs.append((".".join(str(b) for b in sa.addr), int(u.OnLinkPrefixLength)))
            uni = u.Next
        kind = {6: ETHERNET, 71: WIFI, 24: LOOPBACK}.get(int(a.IfType), OTHER)
        speed = int(a.ReceiveLinkSpeed or a.TransmitLinkSpeed)
        if speed in (0, 0xFFFFFFFFFFFFFFFF) or kind == LOOPBACK:
            speed = None                 # bilinmiyor; geri döngünün "hızı" anlamsız
        out.append(NetInterface(a.FriendlyName or (a.AdapterName or b"").decode(errors="replace"),
                                a.Description or "", kind, int(a.OperStatus) == 1, speed,
                                int(a.Mtu) if a.Mtu not in (0, 0xFFFFFFFF) else None, tuple(addrs)))
        node = a.Next
    return out


# ---------------------------------------------------------------------------------------------------- Linux
def _linux_interfaces() -> list[NetInterface]:
    base = Path("/sys/class/net")
    addrs = {i.name: i.addresses for i in _qt_interfaces()}
    out = []
    for path in sorted(base.iterdir()):
        name = path.name

        def read(field: str) -> str | None:
            try:
                return (path / field).read_text().strip()
            except OSError:
                return None

        speed = read("speed")
        mbps = int(speed) if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None
        kind = LOOPBACK if name == "lo" else WIFI if (path / "wireless").exists() else \
            ETHERNET if read("type") == "1" else OTHER
        mtu = read("mtu")
        out.append(NetInterface(name, name, kind, read("operstate") in ("up", "unknown"),
                                mbps * 1_000_000 if mbps else None, int(mtu) if mtu and mtu.isdigit() else None,
                                addrs.get(name, ())))
    return out


def _qt_interfaces() -> list[NetInterface]:
    """QtNetwork ile ad / adres / MTU (bağlantı hızı yok). Qt yoksa boş."""
    try:
        from PyQt6.QtNetwork import QAbstractSocket, QNetworkInterface
    except ImportError:
        return []
    out = []
    for iface in QNetworkInterface.allInterfaces():
        addrs = tuple((e.ip().toString(), e.prefixLength()) for e in iface.addressEntries()
                      if e.ip().protocol() == QAbstractSocket.NetworkLayerProtocol.IPv4Protocol)
        t = iface.type()
        kind = {QNetworkInterface.InterfaceType.Ethernet: ETHERNET, QNetworkInterface.InterfaceType.Wifi: WIFI,
                QNetworkInterface.InterfaceType.Loopback: LOOPBACK}.get(t, OTHER)
        up = bool(iface.flags() & QNetworkInterface.InterfaceFlag.IsRunning)
        out.append(NetInterface(iface.humanReadableName(), iface.name(), kind, up, None,
                                iface.maximumTransmissionUnit() or None, addrs))
    return out
