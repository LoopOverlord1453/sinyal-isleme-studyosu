export const meta = {
  name: 'denetim',
  description: 'Signal Processing Toolbox tam denetimi: 8 uzman ajan paralel denetler, her bulgu 2 bağımsız şüpheciyle doğrulanır, birleşik rapor döner',
  whenToUse: 'Büyük değişikliklerden sonra ya da "her şeyi denetle / doğrula" dendiğinde. args.focus ile odak verilebilir (ör. "filtre tasarımcısı").',
  phases: [
    { title: 'Denetim', detail: '8 uzman ajan paralel, salt okunur' },
    { title: 'Doğrulama', detail: 'her bulgu 2 bağımsız şüpheciyle yeniden üretilir' },
    { title: 'Rapor', detail: 'önem sıralı birleşik rapor' },
  ],
}

// Rol tanımları .claude/agents/*.md içinde. Ajanlar proje dosyalarını DEĞİŞTİRMEZ; bulguları kanıtıyla
// raporlar. Düzeltmeyi ana oturum yapar ve gerekirse bu iş akışını tekrar çalıştırır.
const AUDITORS = [
  { key: 'dsp-dogrulayici', label: 'DSP' },
  { key: 'fuzyon-dogrulayici', label: 'Füzyon' },
  { key: 'arayuz-test-uzmani', label: 'Arayüz' },
  { key: 'protokol-ag-denetcisi', label: 'Protokol/Ağ' },
  { key: 'ux-renk-denetcisi', label: 'UX/Renk' },
  { key: 'kod-gozden-gecirici', label: 'Kod' },
  { key: 'c-kod-dogrulayici', label: 'C kodu' },
  { key: 'sinyal-uretici-dogrulayici', label: 'Sinyal Üreteci' },
]

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['KRİTİK', 'YÜKSEK', 'ORTA', 'DÜŞÜK'] },
          location: { type: 'string', description: 'dosya:satır, görünüm ya da doküman bölümü' },
          title: { type: 'string' },
          expected: { type: 'string' },
          observed: { type: 'string' },
          reproduce: { type: 'string', description: 'yeniden üretme komutu / adımları' },
          fix: { type: 'string' },
        },
        required: ['severity', 'location', 'title', 'observed', 'reproduce'],
      },
    },
    verified_ok: { type: 'array', items: { type: 'string' }, description: 'sorunsuz olduğu doğrulanan iddialar' },
    not_covered: { type: 'array', items: { type: 'string' } },
  },
  required: ['findings', 'verified_ok'],
}

const VERDICT = {
  type: 'object',
  properties: {
    real: { type: 'boolean', description: 'bulgu gerçek ve yeniden üretildi mi' },
    reason: { type: 'string', description: 'çalıştırılan komut ve gözlenen çıktı' },
  },
  required: ['real', 'reason'],
}

const COMMON = 'Proje kökü: Signal_Processing_Toolbox (Windows). Python: .venv\\Scripts\\python ' +
  '(python komutunu kullanma). Proje dosyalarını DEĞİŞTİRME; betikleri %TEMP%\\spt_denetim\\ altında ' +
  'kendi klasörüne yaz. Arayüz denemelerinde QT_QPA_PLATFORM=offscreen ve SPT_SETTINGS_FILE=<geçici ini>. ' +
  'Kullanıcının çalışan uygulamasına ve süreçlerine dokunma; UDP 50505 yerine boş port kullan.'
const focus = args && args.focus ? ` Bu turda özellikle şuna odaklan: ${args.focus}.` : ''

const perAuditor = await pipeline(
  AUDITORS,
  a => agent(
    `Rol tanımın .claude/agents/${a.key}.md dosyasında: önce oku ve harfiyen uygula. ${COMMON}${focus} ` +
    'Her bulguyu çalıştırdığın bir komut ve çıktısıyla destekle; uydurma. Sonucu şemaya göre döndür.',
    { label: `Denetim · ${a.label}`, phase: 'Denetim', schema: FINDINGS },
  ),
  (report, a) => parallel((report ? report.findings : []).map((f, i) => () =>
    parallel([1, 2].map(k => () => agent(
      `Aşağıdaki bulguyu ÇÜRÜTMEYE çalış (şüpheci ${k}). Bağımsız olarak yeniden üretmeyi dene; yeniden ` +
      `üretemiyorsan ya da beklenen davranış zaten doğruysa real=false ver. ${COMMON}\n` +
      `Denetçi: ${a.label}\nBulgu: ${JSON.stringify(f)}`,
      { label: `Doğrula · ${a.label} #${i + 1}.${k}`, phase: 'Doğrulama', schema: VERDICT },
    ))).then(votes => ({ ...f, auditor: a.label, votes: votes.filter(Boolean) })),
  )).then(findings => ({ auditor: a.label, findings, verified_ok: report ? report.verified_ok : [],
                         not_covered: report ? (report.not_covered || []) : [] })),
)

phase('Rapor')
const reports = perAuditor.filter(Boolean)
const all = reports.flatMap(r => r.findings)
const rank = { 'KRİTİK': 0, 'YÜKSEK': 1, 'ORTA': 2, 'DÜŞÜK': 3 }
const bySeverity = (x, y) => rank[x.severity] - rank[y.severity]
const confirmed = all.filter(f => f.votes.length && f.votes.every(v => v.real)).sort(bySeverity)
const disputed = all.filter(f => f.votes.some(v => v.real) && !f.votes.every(v => v.real)).sort(bySeverity)
const rejected = all.filter(f => !f.votes.some(v => v.real))
log(`${all.length} bulgu: ${confirmed.length} doğrulandı, ${disputed.length} tartışmalı, ${rejected.length} reddedildi`)
const missing = AUDITORS.filter(a => !reports.some(r => r.auditor === a.label)).map(a => a.label)
if (missing.length) log(`Rapor dönmeyen denetçiler: ${missing.join(', ')}`)

return {
  confirmed,
  disputed,
  rejected: rejected.map(f => ({ auditor: f.auditor, title: f.title, location: f.location,
                                 why: f.votes.map(v => v.reason) })),
  verified_ok: reports.map(r => ({ auditor: r.auditor, items: r.verified_ok })),
  not_covered: reports.map(r => ({ auditor: r.auditor, items: r.not_covered })),
  missing_auditors: missing,
}
