/* Sinyal İşleme Stüdyosu — arayüz taslakları için ortak betik.
 *
 * Tema geçişi (açık / koyu), ikonlar, örnek sinyal (01_sinus_50Hz_harmonik_2kanal_int16_10kHz kaydını
 * taklit eder), FFT ve canvas grafikleri. Her taslak bu dosyayı <head> içinde yükler (tema boyamadan önce
 * uygulanır) ve sayfa sonunda SPT.baslat({...}) çağırır. Renkler taslağın CSS değişkenlerinden okunur.
 */
(function () {
  'use strict';

  // ------------------------------------------------------------------ Tema
  const ANAHTAR = 'spt-taslak-tema';
  const parametre = new URLSearchParams(location.search);
  const ONIZLEME = parametre.has('onizleme') || window.self !== window.top;

  function ilkTema() {
    const p = parametre.get('tema');
    if (p === 'light' || p === 'dark') return p;
    try {
      const s = localStorage.getItem(ANAHTAR);
      if (s === 'light' || s === 'dark') return s;
    } catch (e) { /* depolama kapalı olabilir */ }
    return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  document.documentElement.dataset.theme = ilkTema();
  if (ONIZLEME) document.documentElement.classList.add('onizleme');

  // ------------------------------------------------------------------ İkonlar (24×24, çizgi)
  const IKON = {
    arama: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    canli: '<path d="M3 3v18h18"/><path d="m7 15 4-5 3 3 5-6"/>',
    nabiz: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    kar: '<path d="M12 2v20M3.3 7l17.4 10M20.7 7 3.3 17"/><path d="m9 4 3 2 3-2M9 20l3-2 3 2M4 10.5l1.5-3.2L2.5 6M20 10.5l-1.5-3.2L21.5 6M4 13.5l1.5 3.2-3 1.3M20 13.5l-1.5 3.2 3 1.3"/>',
    sinus: '<path d="M2 12c1.7-5.3 3.3-8 5-8s3.3 2.7 5 8 3.3 8 5 8 3.3-2.7 5-8"/>',
    filtre: '<path d="M3 4h18l-7 8.5V19l-4 2v-8.5z"/>',
    pusula: '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5z"/>',
    hedef: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    anten: '<circle cx="12" cy="12" r="2"/><path d="M8.5 8.5a5 5 0 0 0 0 7M15.5 8.5a5 5 0 0 1 0 7M5.6 5.6a9 9 0 0 0 0 12.8M18.4 5.6a9 9 0 0 1 0 12.8"/>',
    oynat: '<path d="M7 4.5v15l12-7.5z"/>',
    duraklat: '<path d="M8 5v14M16 5v14"/>',
    dur: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    kayit: '<circle cx="12" cy="12" r="6"/>',
    klasor: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    dosya: '<path d="M6 3h8l5 5v13H6z"/><path d="M14 3v5h5"/>',
    cip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    ayar: '<path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/>',
    gunes: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    ay: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
    buyut: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    tablo: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M9 4v16"/>',
    ayir: '<path d="M14 4h6v6M20 4l-8 8M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4"/>',
    kapat: '<path d="M6 6l12 12M18 6 6 18"/>',
    asagi: '<path d="m6 9 6 6 6-6"/>',
    sag: '<path d="m9 6 6 6-6 6"/>',
    katman: '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    zincir: '<circle cx="6" cy="18" r="2"/><circle cx="18" cy="6" r="2"/><path d="M8 18h6a3 3 0 0 0 0-6h-4a3 3 0 0 1 0-6h6"/>',
    liste: '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4.5 6h.01M4.5 12h.01M4.5 18h.01"/>',
    terminal: '<path d="m5 7 5 5-5 5M12 17h7"/>',
    kod: '<path d="m8 7-5 5 5 5M16 7l5 5-5 5"/>',
    disa: '<path d="M12 15V3M7 8l5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
    cop: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    yukari_ok: '<path d="M12 19V5M6 11l6-6 6 6"/>',
    asagi_ok: '<path d="M12 5v14M6 13l6 6 6-6"/>',
    kalem: '<path d="M4 20h4L19 9l-4-4L4 16z"/>',
    kaydet: '<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M7 3v5h8V3M7 21v-7h10v7"/>',
    arti: '<path d="M12 5v14M5 12h14"/>',
    bilgi: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
    onay: '<path d="m5 12 5 5 9-10"/>',
    onay_daire: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    uyari: '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17v.01"/>',
    imlec: '<circle cx="12" cy="12" r="8"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/>',
    goz: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    yakinlas: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5M8 11h6M11 8v6"/>',
    tasi: '<path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
    dongu: '<path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"/>',
    indir: '<path d="M12 3v12M7 10l5 5 5-5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
    tetik: '<path d="M3 17h7V7h11"/><path d="m7.5 12.5 2.5-2.5 2.5 2.5"/>',
    yan_panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
    alt_panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 14h18"/>',
    sag_panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
    noktalar: '<path d="M5 12h.01M12 12h.01M19 12h.01"/>',
    izgara: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    kitap: '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5M19 19v2H6"/>',
    olcum: '<path d="M3 17 17 3l4 4L7 21z"/><path d="m7 13 2 2M10 10l2 2M13 7l2 2"/>',
    spektrum: '<path d="M4 20V10M8 20V4M12 20v-8M16 20V8M20 20v-5"/>',
    kopyala: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
    sabitle: '<path d="M12 17v5M8 3h8l-1 6 3 4H6l3-4z"/>',
    sinyal_ag: '<path d="M2 20h.01M7 20v-4M12 20v-8M17 20V8M22 4v16"/>',
  };
  const DOLU = new Set(['oynat', 'kayit']);

  function ikonSvg(ad) {
    const govde = IKON[ad] || IKON.noktalar;
    const dolgu = DOLU.has(ad) ? 'currentColor' : 'none';
    return `<svg class="ikon" viewBox="0 0 24 24" fill="${dolgu}" stroke="currentColor" stroke-width="1.75" ` +
      `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${govde}</svg>`;
  }

  function ikonlariYerlestir(kok) {
    (kok || document).querySelectorAll('[data-ikon]').forEach((el) => {
      if (el.dataset.ikonCizildi === el.dataset.ikon) return;
      el.innerHTML = ikonSvg(el.dataset.ikon);
      el.dataset.ikonCizildi = el.dataset.ikon;
    });
  }

  // ------------------------------------------------------------------ Örnek sinyal
  const FS = 10000;          // Hz
  const SURE = 10;           // s, kayıt uzunluğu
  const N_KAYIT = FS * SURE;
  const KANALLAR = ['faz_A', 'faz_B'];
  const W50 = 2 * Math.PI * 50;

  function karisim(n, k) {
    let x = (Math.imul(n, 374761393) + Math.imul(k, 668265263)) | 0;
    x = Math.imul(x ^ (x >>> 13), 1274126177);
    x ^= x >>> 16;
    return (x >>> 0) / 4294967296;
  }

  function gurultu(n, k) {
    // üç tekdüze toplamı ≈ normal, σ ≈ 1
    return (karisim(n, k) + karisim(n, k + 7) + karisim(n, k + 13) - 1.5) * 2;
  }

  function ornek(k, n) {
    const m = ((n % N_KAYIT) + N_KAYIT) % N_KAYIT;
    const t = m / FS;
    if (k === 0) {
      return Math.sin(W50 * t) + 0.08 * Math.sin(3 * W50 * t) +
        0.35 * Math.sin(2 * Math.PI * 1230 * t) + 0.003 * gurultu(m, 1);
    }
    const f = W50 * t - 2.1;
    return 0.8 * Math.sin(f) + 0.05 * Math.sin(5 * f) +
      0.06 * Math.sin(2 * Math.PI * 3110 * t) + 0.003 * gurultu(m, 2);
  }

  // ------------------------------------------------------------------ FFT
  const N_FFT = 8192;
  const HANN = new Float64Array(N_FFT);
  let hannToplam = 0;
  for (let i = 0; i < N_FFT; i++) {
    HANN[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N_FFT - 1));
    hannToplam += HANN[i];
  }

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let g = re[i]; re[i] = re[j]; re[j] = g;
        g = im[i]; im[i] = im[j]; im[j] = g;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const aci = (-2 * Math.PI) / len;
      const wr = Math.cos(aci), wi = Math.sin(aci);
      const yarim = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < yarim; j++) {
          const a = i + j, b = a + yarim;
          const br = re[b] * cr - im[b] * ci;
          const bi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - br; im[b] = im[a] - bi;
          re[a] += br; im[a] += bi;
          const yeni = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = yeni;
        }
      }
    }
  }

  function spektrum(k, nSon) {
    const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
    const n0 = nSon - N_FFT;
    for (let i = 0; i < N_FFT; i++) re[i] = ornek(k, n0 + i) * HANN[i];
    fft(re, im);
    const out = new Float32Array(N_FFT / 2 + 1);
    for (let i = 0; i <= N_FFT / 2; i++) {
      const genlik = (2 * Math.hypot(re[i], im[i])) / hannToplam;
      out[i] = 20 * Math.log10(Math.max(genlik, 1e-9));
    }
    return out;
  }

  // ------------------------------------------------------------------ Yardımcılar
  const EKSI = '−';

  function sayi(v, ondalik) {
    const s = Math.abs(v) < 1e-12 ? (0).toFixed(ondalik) : v.toFixed(ondalik);
    return s.replace('-', EKSI);
  }

  function guzelAdim(aralik, hedef) {
    const ham = aralik / hedef;
    const us = Math.pow(10, Math.floor(Math.log10(ham)));
    const m = ham / us;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * us;
  }

  function ondalikSayisi(adim) {
    return Math.max(0, -Math.floor(Math.log10(adim) + 1e-9));
  }

  function yuvarlakDikdortgen(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // ------------------------------------------------------------------ Motor durumu
  const motor = {
    t: 4.0,
    oynuyor: !matchMedia('(prefers-reduced-motion: reduce)').matches,
    pencere: 0.1,
    gizli: new Set(),
    grafikler: [],
    miniler: [],
    spektrumlar: [null, null],
    sonSpektrum: -1,
    sonOkuma: 0,
    sonMini: 0,
    pps: [],
    imlecler: null,
  };

  // ------------------------------------------------------------------ Grafik
  class Grafik {
    constructor(canvas, ayar) {
      this.c = canvas;
      this.ctx = canvas.getContext('2d');
      this.a = Object.assign({
        tur: 'zaman',          // 'zaman' | 'spektrum'
        stil: 'normal',        // 'normal' | 'osiloskop'
        eksenBasligi: null,    // {x, y}
        y: null,
        xBicim: 'mutlak',      // 'mutlak' (s) | 'goreli' (ms, şimdi = 0)
        tepeler: false,
        kanallar: [{ olcek: 1, ofset: 0 }, { olcek: 1, ofset: 0 }],
        bolme: { x: 10, y: 8 },
        imlecler: null,        // [0.25, 0.45] → pencere oranı
        cizgi: 1.5,
        kenar: null,
      }, ayar);
      this.fare = null;
      this.veri = null;
      this.renkleriOku();
      new ResizeObserver(() => this.boyutla()).observe(canvas);
      canvas.addEventListener('pointermove', (e) => {
        const r = canvas.getBoundingClientRect();
        this.fare = { x: e.clientX - r.left, y: e.clientY - r.top };
        if (!motor.oynuyor) this.ciz();
      });
      canvas.addEventListener('pointerleave', () => {
        this.fare = null;
        if (!motor.oynuyor) this.ciz();
      });
    }

    renkleriOku() {
      const s = getComputedStyle(this.c);
      const v = (n, y) => s.getPropertyValue(n).trim() || y;
      this.r = {
        zemin: v('--grafik-zemin', 'transparent'),
        izgara: v('--grafik-izgara', '#ddd'),
        eksen: v('--grafik-eksen', '#999'),
        metin: v('--grafik-metin', '#666'),
        baslik: v('--grafik-baslik', v('--grafik-metin', '#666')),
        imlec: v('--grafik-imlec', '#888'),
        vurgu: v('--grafik-vurgu', '#e80'),
        ipZemin: v('--grafik-ipucu-zemin', '#fff'),
        ipKenar: v('--grafik-ipucu-kenar', '#ccc'),
        ipMetin: v('--grafik-ipucu-metin', '#111'),
        ipSoluk: v('--grafik-ipucu-soluk', '#666'),
        seri: [v('--seri-1', '#d95926'), v('--seri-2', '#3987e5')],
        font: v('--grafik-font', '11px sans-serif'),
        fontMono: v('--grafik-font-mono', '11px monospace'),
      };
    }

    boyutla() {
      const dpr = window.devicePixelRatio || 1;
      this.w = this.c.clientWidth;
      this.h = this.c.clientHeight;
      if (!this.w || !this.h) return;
      this.c.width = Math.round(this.w * dpr);
      this.c.height = Math.round(this.h * dpr);
      this.dpr = dpr;
      this.ciz();
    }

    alan() {
      const k = this.a.kenar;
      if (k) return { x0: k[3], x1: this.w - k[1], y0: k[0], y1: this.h - k[2] };
      if (this.a.stil === 'osiloskop') return { x0: 22, x1: this.w - 18, y0: 8, y1: this.h - 8 };
      const bas = this.a.eksenBasligi;
      return { x0: bas ? 58 : 44, x1: this.w - 14, y0: 12, y1: this.h - (bas ? 42 : 24) };
    }

    ciz() {
      if (!this.w || !this.h || !this.veri) return;
      const ctx = this.ctx;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      if (this.r.zemin && this.r.zemin !== 'transparent') {
        ctx.fillStyle = this.r.zemin;
        ctx.fillRect(0, 0, this.w, this.h);
      }
      if (this.a.stil === 'osiloskop') this.cizOsiloskop();
      else this.cizNormal();
    }

    // ---- standart eksenli grafik
    cizNormal() {
      const ctx = this.ctx, a = this.alan(), r = this.r, v = this.veri;
      const spek = this.a.tur === 'spektrum';
      const xmin = v.xmin, xmax = v.xmax;
      const ya = this.a.y || (spek ? { min: -120, max: 0, adim: 20 } : { min: -1.6, max: 1.6, adim: 0.5 });
      const px = (x) => a.x0 + ((x - xmin) / (xmax - xmin)) * (a.x1 - a.x0);
      const py = (y) => a.y1 - ((y - ya.min) / (ya.max - ya.min)) * (a.y1 - a.y0);
      ctx.font = r.font;
      ctx.lineWidth = 1;

      // yatay ızgara + y etiketleri
      const yOnd = ondalikSayisi(ya.adim);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let y = Math.ceil(ya.min / ya.adim) * ya.adim; y <= ya.max + 1e-9; y += ya.adim) {
        const yy = Math.round(py(y)) + 0.5;
        ctx.strokeStyle = Math.abs(y) < 1e-9 && !spek ? r.eksen : r.izgara;
        ctx.beginPath(); ctx.moveTo(a.x0, yy); ctx.lineTo(a.x1, yy); ctx.stroke();
        ctx.fillStyle = r.metin;
        ctx.fillText(sayi(y, yOnd), a.x0 - 8, yy);
      }

      // dikey ızgara + x etiketleri
      const adim = guzelAdim(xmax - xmin, Math.max(4, Math.floor((a.x1 - a.x0) / 90)));
      const xOnd = ondalikSayisi(adim);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let x = Math.ceil(xmin / adim - 1e-9) * adim; x <= xmax + 1e-9; x += adim) {
        const xx = Math.round(px(x)) + 0.5;
        ctx.strokeStyle = r.izgara;
        ctx.beginPath(); ctx.moveTo(xx, a.y0); ctx.lineTo(xx, a.y1); ctx.stroke();
        ctx.fillStyle = r.metin;
        ctx.fillText(sayi(x, xOnd), xx, a.y1 + 7);
      }
      ctx.strokeStyle = r.eksen;
      ctx.beginPath(); ctx.moveTo(a.x0, Math.round(a.y1) + 0.5); ctx.lineTo(a.x1, Math.round(a.y1) + 0.5); ctx.stroke();

      // eksen başlıkları
      if (this.a.eksenBasligi) {
        ctx.fillStyle = r.baslik;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(this.a.eksenBasligi.x, (a.x0 + a.x1) / 2, this.h - 8);
        ctx.save();
        ctx.translate(14, (a.y0 + a.y1) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = 'middle';
        ctx.fillText(this.a.eksenBasligi.y, 0, 0);
        ctx.restore();
      }

      // seriler
      ctx.save();
      ctx.beginPath(); ctx.rect(a.x0, a.y0 - 2, a.x1 - a.x0, a.y1 - a.y0 + 4); ctx.clip();
      ctx.lineWidth = this.a.cizgi;
      ctx.lineJoin = 'round';
      v.ys.forEach((ys, k) => {
        if (motor.gizli.has(k) || !ys) return;
        ctx.strokeStyle = r.seri[k];
        ctx.beginPath();
        if (spek) {
          // piksel sütunu başına tepe korumalı seyreltme
          const n = ys.length;
          for (let c = Math.floor(a.x0); c < a.x1 - 1; c++) {
            const b0 = Math.max(0, Math.floor(((c - a.x0) / (a.x1 - a.x0)) * (n - 1)));
            const b1 = Math.min(n - 1, Math.floor(((c + 1 - a.x0) / (a.x1 - a.x0)) * (n - 1)));
            let m = -Infinity;
            for (let b = b0; b <= b1; b++) if (ys[b] > m) m = ys[b];
            const yy = py(Math.max(m, ya.min - 5));
            if (c === Math.floor(a.x0)) ctx.moveTo(c, yy); else ctx.lineTo(c, yy);
          }
        } else {
          const n = ys.length;
          for (let i = 0; i < n; i++) {
            const xx = a.x0 + (i / (n - 1)) * (a.x1 - a.x0);
            if (i === 0) ctx.moveTo(xx, py(ys[i])); else ctx.lineTo(xx, py(ys[i]));
          }
        }
        ctx.stroke();
      });
      ctx.restore();

      if (spek && this.a.tepeler) this.tepeEtiketleri(a, px, py, ya);
      this.ipucu(a, px, py, ya);
    }

    tepeEtiketleri(a, px, py, ya) {
      const ctx = this.ctx, r = this.r, v = this.veri;
      const tepeler = [];
      v.ys.forEach((ys, k) => {
        if (motor.gizli.has(k) || !ys) return;
        for (let i = 3; i < ys.length - 3; i++) {
          if (ys[i] > -45 && ys[i] >= ys[i - 1] && ys[i] >= ys[i + 1] && ys[i] >= ys[i - 3] && ys[i] >= ys[i + 3]) {
            tepeler.push({ k, f: (i / (ys.length - 1)) * v.xmax, db: ys[i] });
          }
        }
      });
      tepeler.sort((p, q) => q.db - p.db);
      const secili = [];
      for (const p of tepeler) {
        if (secili.some((s) => Math.abs(px(s.f) - px(p.f)) < 46)) continue;
        secili.push(p);
        if (secili.length >= 4) break;
      }
      ctx.font = r.font;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      for (const p of secili) {
        const x = px(p.f), y = py(p.db);
        ctx.fillStyle = r.seri[p.k];
        ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x - 4, y - 10); ctx.lineTo(x + 4, y - 10); ctx.closePath(); ctx.fill();
        const f = p.f >= 1 ? `${sayi(p.f, 2)} kHz` : `${Math.round(p.f * 1000)} Hz`;
        ctx.fillStyle = r.metin;
        ctx.fillText(f, Math.min(x + 6, a.x1 - 52), Math.max(y - 9, a.y0 + 6));
      }
    }

    // ---- osiloskop ızgarası (bölme tabanlı)
    cizOsiloskop() {
      const ctx = this.ctx, a = this.alan(), r = this.r, v = this.veri;
      const bx = this.a.bolme.x, by = this.a.bolme.y;
      const dx = (a.x1 - a.x0) / bx, dy = (a.y1 - a.y0) / by;
      const py = (d) => (a.y0 + a.y1) / 2 - d * dy;
      const px = (x) => a.x0 + ((x - v.xmin) / (v.xmax - v.xmin)) * (a.x1 - a.x0);
      const cx = Math.round((a.x0 + a.x1) / 2) + 0.5, cy = Math.round((a.y0 + a.y1) / 2) + 0.5;

      ctx.lineWidth = 1;
      ctx.strokeStyle = r.izgara;
      ctx.setLineDash([1, 3]);
      for (let i = 1; i < bx; i++) {
        const x = Math.round(a.x0 + i * dx) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, a.y0); ctx.lineTo(x, a.y1); ctx.stroke();
      }
      for (let j = 1; j < by; j++) {
        const y = Math.round(a.y0 + j * dy) + 0.5;
        ctx.beginPath(); ctx.moveTo(a.x0, y); ctx.lineTo(a.x1, y); ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.strokeStyle = r.eksen;
      ctx.beginPath(); ctx.moveTo(a.x0, cy); ctx.lineTo(a.x1, cy); ctx.moveTo(cx, a.y0); ctx.lineTo(cx, a.y1); ctx.stroke();
      for (let i = 0; i <= bx * 5; i++) {
        const x = Math.round(a.x0 + (i * dx) / 5) + 0.5;
        const l = i % 5 === 0 ? 5 : 3;
        ctx.beginPath(); ctx.moveTo(x, cy - l); ctx.lineTo(x, cy + l); ctx.stroke();
      }
      for (let j = 0; j <= by * 5; j++) {
        const y = Math.round(a.y0 + (j * dy) / 5) + 0.5;
        const l = j % 5 === 0 ? 5 : 3;
        ctx.beginPath(); ctx.moveTo(cx - l, y); ctx.lineTo(cx + l, y); ctx.stroke();
      }
      ctx.strokeRect(Math.round(a.x0) + 0.5, Math.round(a.y0) + 0.5, Math.round(a.x1 - a.x0), Math.round(a.y1 - a.y0));

      // ölçüm imleçleri
      if (this.a.imlecler) {
        ctx.strokeStyle = r.vurgu;
        ctx.setLineDash([4, 3]);
        ctx.font = r.fontMono;
        ctx.fillStyle = r.vurgu;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        this.a.imlecler.forEach((o, i) => {
          const x = Math.round(a.x0 + o * (a.x1 - a.x0)) + 0.5;
          ctx.beginPath(); ctx.moveTo(x, a.y0); ctx.lineTo(x, a.y1); ctx.stroke();
          ctx.fillText(i === 0 ? 'A' : 'B', x + 4, a.y0 + 4);
        });
        ctx.setLineDash([]);
      }

      // izler
      ctx.save();
      ctx.beginPath(); ctx.rect(a.x0, a.y0, a.x1 - a.x0, a.y1 - a.y0); ctx.clip();
      ctx.lineWidth = this.a.cizgi;
      ctx.lineJoin = 'round';
      v.ys.forEach((ys, k) => {
        if (motor.gizli.has(k) || !ys) return;
        const kn = this.a.kanallar[k];
        ctx.strokeStyle = r.seri[k];
        ctx.beginPath();
        const n = ys.length;
        for (let i = 0; i < n; i++) {
          const xx = a.x0 + (i / (n - 1)) * (a.x1 - a.x0);
          const yy = py(ys[i] / kn.olcek + kn.ofset);
          if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
        }
        ctx.stroke();
      });
      ctx.restore();

      // toprak işaretleri (sol) ve tetik (sağ)
      ctx.font = `600 ${r.fontMono.replace(/^\s*\d+(\.\d+)?px/, '10px')}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      this.a.kanallar.forEach((kn, k) => {
        if (motor.gizli.has(k)) return;
        const y = py(kn.ofset);
        ctx.fillStyle = r.seri[k];
        ctx.beginPath();
        ctx.moveTo(a.x0 - 1, y); ctx.lineTo(a.x0 - 7, y - 7); ctx.lineTo(a.x0 - 20, y - 7);
        ctx.lineTo(a.x0 - 20, y + 7); ctx.lineTo(a.x0 - 7, y + 7); ctx.closePath(); ctx.fill();
        ctx.fillStyle = r.zemin && r.zemin !== 'transparent' ? r.zemin : '#000';
        ctx.fillText(String(k + 1), a.x0 - 12, y + 0.5);
      });
      const ty = py(this.a.kanallar[0].ofset);
      ctx.fillStyle = r.vurgu;
      ctx.beginPath();
      ctx.moveTo(a.x1 + 1, ty); ctx.lineTo(a.x1 + 7, ty - 7); ctx.lineTo(a.x1 + 17, ty - 7);
      ctx.lineTo(a.x1 + 17, ty + 7); ctx.lineTo(a.x1 + 7, ty + 7); ctx.closePath(); ctx.fill();
      ctx.fillStyle = r.zemin && r.zemin !== 'transparent' ? r.zemin : '#000';
      ctx.fillText('T', a.x1 + 11, ty + 0.5);

      const kn = this.a.kanallar;
      this.ipucu(a, px, (val, k) => py(val / kn[k].olcek + kn[k].ofset), null);
    }

    // ---- fare ipucu (artı imleç + değerler)
    ipucu(a, px, py) {
      const f = this.fare;
      if (!f || f.x < a.x0 || f.x > a.x1 || f.y < a.y0 || f.y > a.y1) return;
      const ctx = this.ctx, r = this.r, v = this.veri;
      const spek = this.a.tur === 'spektrum';
      const oran = (f.x - a.x0) / (a.x1 - a.x0);
      const satirlar = [];
      const noktalar = [];
      let baslik;
      if (spek) {
        const n = v.ys[0].length;
        const i = Math.round(oran * (n - 1));
        const fk = (i / (n - 1)) * v.xmax;
        baslik = fk >= 1 ? `f = ${sayi(fk, 3)} kHz` : `f = ${sayi(fk * 1000, 1)} Hz`;
        v.ys.forEach((ys, k) => {
          if (motor.gizli.has(k) || !ys) return;
          let m = -Infinity;
          for (let b = Math.max(0, i - 2); b <= Math.min(n - 1, i + 2); b++) m = Math.max(m, ys[b]);
          satirlar.push({ k, metin: `${sayi(m, 1)} dB` });
          noktalar.push({ k, y: py(m) });
        });
      } else {
        const n = v.ys[0].length;
        const i = Math.round(oran * (n - 1));
        const t = v.xmin + oran * (v.xmax - v.xmin);
        baslik = this.a.xBicim === 'goreli' ? `t = ${sayi(t, 1)} ms` : `t = ${sayi(t, 4)} s`;
        v.ys.forEach((ys, k) => {
          if (motor.gizli.has(k) || !ys) return;
          satirlar.push({ k, metin: `${sayi(ys[i], 3)} V` });
          noktalar.push({ k, y: py(ys[i], k) });
        });
      }

      const x = Math.round(f.x) + 0.5;
      ctx.strokeStyle = r.imlec;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, a.y0); ctx.lineTo(x, a.y1); ctx.stroke();
      ctx.setLineDash([]);
      for (const p of noktalar) {
        ctx.fillStyle = r.seri[p.k];
        ctx.strokeStyle = r.zemin && r.zemin !== 'transparent' ? r.zemin : r.ipZemin;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, p.y, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }

      ctx.font = r.fontMono;
      const satirH = 17;
      const genislik = Math.max(ctx.measureText(baslik).width,
        ...satirlar.map((s) => ctx.measureText(`${KANALLAR[s.k]}  ${s.metin}`).width + 18)) + 20;
      const yukseklik = 12 + satirH * (satirlar.length + 1);
      let bx = f.x + 14;
      if (bx + genislik > a.x1) bx = f.x - 14 - genislik;
      const by = Math.min(Math.max(a.y0 + 4, f.y - yukseklik / 2), a.y1 - yukseklik - 4);
      ctx.fillStyle = r.ipZemin;
      ctx.strokeStyle = r.ipKenar;
      ctx.lineWidth = 1;
      yuvarlakDikdortgen(ctx, Math.round(bx) + 0.5, Math.round(by) + 0.5, genislik, yukseklik, 6);
      ctx.fill(); ctx.stroke();
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = r.ipSoluk;
      ctx.fillText(baslik, bx + 10, by + 6 + satirH / 2);
      satirlar.forEach((s, j) => {
        const yy = by + 6 + satirH * (j + 1) + satirH / 2;
        ctx.fillStyle = r.seri[s.k];
        ctx.fillRect(bx + 10, yy - 1, 10, 2.5);
        ctx.fillStyle = r.ipMetin;
        ctx.fillText(`${KANALLAR[s.k]}  ${s.metin}`, bx + 28, yy);
      });
    }
  }

  // ------------------------------------------------------------------ Küçük çizgi grafik (KPI)
  class MiniGrafik {
    constructor(canvas, ayar) {
      this.c = canvas;
      this.ctx = canvas.getContext('2d');
      this.a = Object.assign({ renk: '--vurgu', min: 20, max: 32 }, ayar);
      this.degerler = [];
      new ResizeObserver(() => this.boyutla()).observe(canvas);
    }
    boyutla() {
      const dpr = window.devicePixelRatio || 1;
      this.w = this.c.clientWidth; this.h = this.c.clientHeight;
      this.c.width = Math.round(this.w * dpr); this.c.height = Math.round(this.h * dpr);
      this.dpr = dpr;
      this.ciz();
    }
    ciz() {
      if (!this.w || this.degerler.length < 2) return;
      const ctx = this.ctx, d = this.degerler, { min, max } = this.a;
      const renk = getComputedStyle(this.c).getPropertyValue(this.a.renk).trim() || '#888';
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      const px = (i) => 2 + (i / (MINI_UZUNLUK - 1)) * (this.w - 6);
      const py = (v) => this.h - 3 - ((v - min) / (max - min)) * (this.h - 6);
      const bas = MINI_UZUNLUK - d.length;
      ctx.strokeStyle = renk;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      d.forEach((v, i) => (i ? ctx.lineTo(px(bas + i), py(v)) : ctx.moveTo(px(bas + i), py(v))));
      ctx.stroke();
      ctx.fillStyle = renk;
      ctx.beginPath(); ctx.arc(px(bas + d.length - 1), py(d[d.length - 1]), 2.5, 0, Math.PI * 2); ctx.fill();
    }
  }
  const MINI_UZUNLUK = 40;

  // ------------------------------------------------------------------ Döngü
  function pencereVerisi() {
    const n = Math.round(motor.pencere * FS);
    const nSon = Math.floor(motor.t * FS);
    const ys = [new Float32Array(n), new Float32Array(n)];
    for (let k = 0; k < 2; k++) for (let i = 0; i < n; i++) ys[k][i] = ornek(k, nSon - n + i);
    return { nSon, ys };
  }

  function guncelle(simdi, zorla) {
    const { nSon, ys } = pencereVerisi();
    const tSon = nSon / FS;

    if (zorla || simdi - motor.sonSpektrum > 200) {
      motor.spektrumlar = [spektrum(0, nSon), spektrum(1, nSon)];
      motor.sonSpektrum = simdi;
    }

    for (const g of motor.grafikler) {
      if (g.a.tur === 'spektrum') {
        g.veri = { xmin: 0, xmax: FS / 2000, ys: motor.spektrumlar };
      } else if (g.a.xBicim === 'goreli') {
        g.veri = { xmin: -motor.pencere * 1000, xmax: 0, ys };
      } else {
        g.veri = { xmin: tSon - motor.pencere, xmax: tSon, ys };
      }
      g.ciz();
    }

    if (zorla || simdi - motor.sonOkuma > 250) {
      motor.sonOkuma = simdi;
      okumalariYaz(ys, tSon);
    }
    if (zorla || simdi - motor.sonMini > 600) {
      motor.sonMini = simdi;
      const pps = motor.oynuyor ? 27 + Math.round((karisim(Math.floor(simdi), 3) - 0.5) * 2.2) : 0;
      for (const m of motor.miniler) {
        const deger = m.a.kaynak === 'kbps' ? pps * 1.422 : pps;
        m.degerler.push(deger);
        if (m.degerler.length > MINI_UZUNLUK) m.degerler.shift();
        m.ciz();
      }
    }
  }

  function yaz(anahtar, metin) {
    document.querySelectorAll(`[data-canli="${anahtar}"]`).forEach((el) => {
      if (el.textContent !== metin) el.textContent = metin;
    });
  }

  function sureMetni(t) {
    const dk = Math.floor(t / 60), sn = t - dk * 60;
    return `${String(dk).padStart(2, '0')}:${sn.toFixed(1).padStart(4, '0')}`;
  }

  function okumalariYaz(ys, tSon) {
    const tKayit = ((tSon % SURE) + SURE) % SURE;
    yaz('konum', sureMetni(tKayit));
    yaz('sure', sureMetni(SURE));
    document.querySelectorAll('[data-canli="ilerleme"]').forEach((el) =>
      el.style.setProperty('--ilerleme', `${((tKayit / SURE) * 100).toFixed(2)}%`));
    const j = (s) => (karisim(Math.floor(tSon * 4), s) - 0.5);
    const pps = motor.oynuyor ? 27 + Math.round(j(5) * 2.2) : 0;
    yaz('pps', String(pps));
    yaz('kbps', motor.oynuyor ? (38.4 + j(6) * 1.2).toFixed(1) : '0.0');
    yaz('paket', String(Math.floor(tKayit * 27.0) + 1).replace(/\B(?=(\d{3})+(?!\d))/g, '.'));
    yaz('ornek', String(Math.floor(tKayit * FS)).replace(/\B(?=(\d{3})+(?!\d))/g, '.'));
    yaz('kayip', '0');
    for (let k = 0; k < 2; k++) {
      const d = ys[k];
      let mn = Infinity, mx = -Infinity, kare = 0, top = 0;
      for (let i = 0; i < d.length; i++) {
        const x = d[i];
        if (x < mn) mn = x;
        if (x > mx) mx = x;
        kare += x * x; top += x;
      }
      yaz(`vpp-${k}`, sayi(mx - mn, 3));
      yaz(`rms-${k}`, sayi(Math.sqrt(kare / d.length), 3));
      yaz(`ort-${k}`, sayi(top / d.length, 3));
      yaz(`min-${k}`, sayi(mn, 3));
      yaz(`maks-${k}`, sayi(mx, 3));
      yaz(`f-${k}`, (50 + j(10 + k) * 0.02).toFixed(2));
      yaz(`thd-${k}`, (k === 0 ? 8.0 + j(20) * 0.1 : 6.25 + j(21) * 0.1).toFixed(2));
      if (motor.imlecler) {
        motor.imlecler.forEach((o, ci) => {
          const i = Math.round(o * (d.length - 1));
          yaz(`imlec${ci ? 'B' : 'A'}-${k}`, sayi(d[i], 3));
        });
        const ia = Math.round(motor.imlecler[0] * (d.length - 1));
        const ib = Math.round(motor.imlecler[1] * (d.length - 1));
        yaz(`imlecD-${k}`, sayi(d[ib] - d[ia], 3));
      }
    }
    if (motor.imlecler) {
      const dt = (motor.imlecler[1] - motor.imlecler[0]) * motor.pencere;
      const t0 = tKayit - motor.pencere;
      yaz('imlecA-t', sayi(t0 + motor.imlecler[0] * motor.pencere, 4));
      yaz('imlecB-t', sayi(t0 + motor.imlecler[1] * motor.pencere, 4));
      yaz('imlec-dt', (dt * 1000).toFixed(2));
      yaz('imlec-f', (1 / dt).toFixed(2));
    }
    document.querySelectorAll('[data-canli="durum"]').forEach((el) => {
      const m = motor.oynuyor ? (el.dataset.metinOynuyor || 'Kayıt oynatılıyor')
        : (el.dataset.metinDonuk || 'Donduruldu');
      if (el.textContent !== m) el.textContent = m;
    });
  }

  let sonKare = 0, oncekiZaman = 0;
  function kare(simdi) {
    if (!oncekiZaman) oncekiZaman = simdi;
    const dt = Math.min(0.1, (simdi - oncekiZaman) / 1000);
    oncekiZaman = simdi;
    if (motor.oynuyor) {
      motor.t += dt;
      if (simdi - sonKare >= 33) {
        sonKare = simdi;
        guncelle(simdi, false);
      }
    }
    requestAnimationFrame(kare);
  }

  // ------------------------------------------------------------------ Etkileşim
  function temaUygula(t, kaydet) {
    document.documentElement.dataset.theme = t;
    if (kaydet) {
      try { localStorage.setItem(ANAHTAR, t); } catch (e) { /* yok say */ }
    }
    document.querySelectorAll('[data-tema-dugme]').forEach((b) => {
      b.setAttribute('aria-pressed', String(t === 'dark'));
      b.setAttribute('title', t === 'dark' ? 'Açık temaya geç' : 'Koyu temaya geç');
      const ik = b.querySelector('[data-tema-ikon]');
      if (ik) { ik.dataset.ikon = t === 'dark' ? 'gunes' : 'ay'; ikonlariYerlestir(b); }
      const et = b.querySelector('[data-tema-etiket]');
      if (et) et.textContent = t === 'dark' ? 'Açık tema' : 'Koyu tema';
    });
    for (const g of motor.grafikler) { g.renkleriOku(); g.ciz(); }
    for (const m of motor.miniler) m.ciz();
  }

  function dondur(donuk) {
    motor.oynuyor = !donuk;
    document.documentElement.classList.toggle('donuk', donuk);
    document.querySelectorAll('[data-dondur]').forEach((b) => {
      b.setAttribute('aria-pressed', String(donuk));
      const ik = b.querySelector('[data-dondur-ikon]');
      if (ik) { ik.dataset.ikon = donuk ? 'oynat' : 'duraklat'; ikonlariYerlestir(b); }
      const et = b.querySelector('[data-dondur-etiket]');
      if (et) et.textContent = donuk ? (b.dataset.etiketDonuk || 'Sürdür') : (b.dataset.etiketCanli || 'Dondur');
    });
    guncelle(performance.now(), true);
  }

  function pencereAyarla(ms) {
    motor.pencere = ms / 1000;
    document.querySelectorAll('[data-canli="pencere"]').forEach((el) => { el.textContent = `${ms} ms`; });
    guncelle(performance.now(), true);
  }

  function secimGruplari() {
    document.querySelectorAll('[data-secim-grubu]').forEach((grup) => {
      grup.addEventListener('click', (e) => {
        const b = e.target.closest('button');
        if (!b || !grup.contains(b) || b.closest('[data-secim-grubu]') !== grup) return;
        grup.querySelectorAll(':scope button').forEach((x) => {
          if (x.closest('[data-secim-grubu]') !== grup) return;
          x.classList.toggle('secili', x === b);
          x.setAttribute('aria-pressed', String(x === b));
        });
        if (b.dataset.pencere) pencereAyarla(Number(b.dataset.pencere));
      });
    });
    document.querySelectorAll('[data-sekmeler]').forEach((kap) => {
      kap.addEventListener('click', (e) => {
        const b = e.target.closest('[data-sekme]');
        if (!b || b.closest('[data-sekmeler]') !== kap) return;
        kap.querySelectorAll('[data-sekme]').forEach((x) => {
          if (x.closest('[data-sekmeler]') !== kap) return;
          const on = x === b;
          x.classList.toggle('secili', on);
          x.setAttribute('aria-selected', String(on));
        });
        kap.querySelectorAll('[data-sekme-panel]').forEach((p) => {
          if (p.closest('[data-sekmeler]') !== kap) return;
          p.hidden = p.dataset.sekmePanel !== b.dataset.sekme;
        });
      });
    });
  }

  function taslakEtiketi(t) {
    if (!t || ONIZLEME) return;
    const el = document.createElement('div');
    el.className = 'spt-taslak-etiketi';
    el.innerHTML = `<b>Taslak ${t.harf}</b><span>${t.ad}</span>` +
      '<a href="index.html">Tüm taslaklar</a><button type="button" aria-label="Etiketi gizle">×</button>';
    el.querySelector('button').addEventListener('click', () => el.remove());
    document.body.appendChild(el);
  }

  const TEMEL_CSS = `
    canvas[data-grafik], canvas[data-mini] { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
    canvas[data-grafik] { cursor: crosshair; touch-action: none; }
    .ikon { width: 1em; height: 1em; display: block; flex: none; }
    [data-ikon] { display: inline-flex; align-items: center; justify-content: center; flex: none; }
    .spt-taslak-etiketi { position: fixed; left: 50%; bottom: 40px; transform: translateX(-50%); z-index: 50;
      display: flex; align-items: center; gap: 10px; padding: 7px 8px 7px 14px; border-radius: 999px;
      background: #17171a; color: #f2f2f3; border: 1px solid #3a3a40; font: 12px/1.2 "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif; }
    .spt-taslak-etiketi span { color: #b9b9c0; }
    .spt-taslak-etiketi a { color: #f2f2f3; text-decoration: underline; text-underline-offset: 2px; }
    .spt-taslak-etiketi button { all: unset; cursor: pointer; width: 22px; height: 22px; border-radius: 50%;
      display: grid; place-items: center; color: #b9b9c0; }
    .spt-taslak-etiketi button:hover { background: #2a2a30; color: #fff; }
    html.onizleme .spt-taslak-etiketi { display: none; }
  `;

  function baslat(ayar) {
    ayar = ayar || {};
    const stil = document.createElement('style');
    stil.textContent = TEMEL_CSS;
    document.head.prepend(stil);

    ikonlariYerlestir();
    if (ayar.pencereMs) motor.pencere = ayar.pencereMs / 1000;
    motor.imlecler = ayar.imlecler || null;

    (ayar.grafikler || []).forEach((g) => {
      const c = typeof g.el === 'string' ? document.querySelector(g.el) : g.el;
      if (!c) return;
      c.setAttribute('data-grafik', '');
      motor.grafikler.push(new Grafik(c, g));
    });
    (ayar.miniler || []).forEach((m) => {
      const c = typeof m.el === 'string' ? document.querySelector(m.el) : m.el;
      if (!c) return;
      c.setAttribute('data-mini', '');
      const mg = new MiniGrafik(c, m);
      for (let i = 0; i < MINI_UZUNLUK; i++) {
        const p = 27 + Math.round((karisim(i, 3) - 0.5) * 2.2);
        mg.degerler.push(m.kaynak === 'kbps' ? p * 1.422 : p);
      }
      motor.miniler.push(mg);
    });

    document.querySelectorAll('[data-tema-dugme]').forEach((b) =>
      b.addEventListener('click', () =>
        temaUygula(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark', true)));
    document.querySelectorAll('[data-dondur]').forEach((b) =>
      b.addEventListener('click', () => dondur(motor.oynuyor)));
    document.querySelectorAll('[data-kanal]').forEach((cb) =>
      cb.addEventListener('change', () => {
        const k = Number(cb.dataset.kanal);
        if (cb.checked) motor.gizli.delete(k); else motor.gizli.add(k);
        document.querySelectorAll(`[data-kanal="${k}"]`).forEach((x) => { x.checked = cb.checked; });
        guncelle(performance.now(), true);
      }));
    window.addEventListener('keydown', (e) => {
      if (e.key === 'F6') { e.preventDefault(); dondur(motor.oynuyor); }
    });
    window.addEventListener('message', (e) => {
      if (e.data && e.data.sptTema) temaUygula(e.data.sptTema, false);
    });
    secimGruplari();
    taslakEtiketi(ayar.taslak);

    temaUygula(document.documentElement.dataset.theme, false);
    dondur(!motor.oynuyor);
    requestAnimationFrame(kare);
  }

  window.SPT = { baslat, temaUygula, dondur, pencereAyarla, ikonlariYerlestir };
})();
