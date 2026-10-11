"""Blok diyagram benzetim motoru (Simulink benzeri; arayüzsüz çekirdek).

Hızlı başlangıç
---------------
    from spt.control.diagram import Model, simulate, linearize, linio, findop

    m = Model("konum denetimi")
    r = m.add("Step", "Referans", step_time=0.0)
    e = m.add("Sum", "Hata", signs="+-")
    c = m.add("PIDController", "PID", P=4.0, I=2.0, D=0.5, N=50.0, limit_output=True, upper=10, lower=-10,
              anti_windup="clamping")
    g = m.add("TransferFcn", "Bitki", num=[1.0], den=[1.0, 3.0, 2.0])
    m.connect(r, e); m.connect(e, c, name="e"); m.connect(c, g, name="u"); m.connect(g, e, dst_port=1)
    m.log_signal(g, name="y")
    res = simulate(m, 10.0, solver="ode45", rtol=1e-6)
    t, y = res.signal("y")
    L = linearize(m, [linio("PID", 0, "looptransfer")])         # çevrim transferi (−C·G)

Modüller
--------
- `block`  : blok sözleşmesi (`Block`, `Param`, `SampleTime`), kayıt (`register_block`, `register_lazy`,
             `block_catalog`, `create_block`).
- `blocks/`: blok kütüphanesi (kaynaklar, sürekli, ayrık, doğrusal olmayan, kontrolcüler, yönlendirme, gözlem,
             portlar / alt sistem, bitki eklentileri).
- `model`  : `Model` (bloklar, bağlantılar, sinyal adları, alt sistemler, doğrulama, `.sptd` kaydet / aç).
- `engine` : `compile_model`, `simulate`, `Simulator` (ode1 / ode4 / ode45, çok hızlı, sıfır geçiş, ilerleme /
             iptal), `SimResult`, `OperatingPoint`.
- `linearize`: `linearize`, `linio`, `LinearSystem`, `findop`, `operspec`.
- `realdata`: gerçek veriyle benzetim ve karşılaştırma (`simulate_with_data`, `fit_percent`).

Eklenti (yeni blok) sözleşmesi
------------------------------
Yeni blok, `Block`'tan türetilip `@register_block` ile kaydedilir:

    from spt.control.diagram import Block, Param, p_ts, register_block

    @register_block
    class Kare(Block):
        type_name = "Square"              # kalıcı tür adı (dosyada saklanır)
        display_name = "Kare"             # Türkçe görünen ad
        category = "nonlinear"            # CATEGORIES anahtarı
        description = "y = u²"
        PARAMS = (Param("k", "float", 1.0, "Kazanç"), p_ts(-1.0))
        def output(self, t, x, xd, u):    # u: giriş portu dizileri listesi; dönüş: çıkış dizileri listesi
            return [self.params["k"] * u[0] ** 2]

Zorunlu / isteğe bağlı yöntemler: `n_inputs`, `n_outputs`, `input_labels`, `output_labels`, `sample_time`
(varsayılan 'Ts' parametresi), `allowed_sample_times`, `feedthrough`, `propagate_widths(in_widths, guess)`,
`setup(in_widths, out_widths)` (burada `nx` / `nxd` belirlenir), `x0`, `xd0`, `reset` (iş belleği), `output`,
`derivatives`, `update`, `major` (ana adımda çıkıştan önce kip güncelleme), `commit` (ana adımda çıkıştan sonra iş
belleği), `project` (durum kırpma), `next_event` (zaman olayı), `zero_crossings` (kip sınırı sinyalleri),
`linearize` (analitik (A, B, C, D); None ise sayısal), `trim_output` (kararlı hâl çıkışı). Sınıf özniteliği
`steady_passthrough = True`: kararlı hâlde çıkış = giriş (taşıma gecikmesi gibi; trim ek bilinmeyen kullanır).
Python nesnesi bellekli kontrolcüler (MPC) için `DiscreteController`, Python'da yazılmış sürekli bitkiler (İHA)
için `ContinuousPlant` taban sınıfları vardır. Başka paketler bloklarını `spt.control.mpc.diagram_blocks` ya da
`spt.control.uav.diagram_blocks` modüllerinde tanımlarsa kütüphane bunları kendiliğinden yükler; ya da
`register_lazy("MPCController", "paket.modul:Sinif", display_name="MPC kontrolcü")` ile tembel kayıt yapılır.
"""
from spt.control.diagram.block import (CATEGORIES, CONSTANT_TS, CONTINUOUS_TS, INHERITED_TS, Block, Param,
                                       SampleTime, SimContext, SourceBlock, block_catalog, block_class, block_types,
                                       create_block, make_sample_time, p_ts, register_block, register_lazy)
from spt.control.diagram.blocks.continuous import LTIBlock, lti_block
from spt.control.diagram.blocks.plants import ContinuousPlant, DiscreteController
from spt.control.diagram.engine import (SOLVERS, CompiledModel, OperatingPoint, ScopeData, SimResult, Simulator,
                                        compile_model, simulate)
from spt.control.diagram.linearize import (POINT_TYPES, LinearSystem, LinPoint, OperSpec, TrimResult, findop,
                                           linearize, linio, operspec)
from spt.control.diagram.model import FILE_SUFFIX, Connection, Model, ValidationReport
from spt.control.diagram.realdata import DataComparison, fit_percent, simulate_with_data

__all__ = [
    "Block", "Param", "SampleTime", "SimContext", "SourceBlock", "CATEGORIES", "CONSTANT_TS", "CONTINUOUS_TS",
    "INHERITED_TS", "block_catalog", "block_class", "block_types", "create_block", "make_sample_time", "p_ts",
    "register_block", "register_lazy", "LTIBlock", "lti_block", "ContinuousPlant", "DiscreteController",
    "SOLVERS", "CompiledModel", "OperatingPoint", "ScopeData", "SimResult", "Simulator", "compile_model", "simulate",
    "POINT_TYPES", "LinearSystem", "LinPoint", "OperSpec", "TrimResult", "findop", "linearize", "linio", "operspec",
    "FILE_SUFFIX", "Connection", "Model", "ValidationReport",
    "DataComparison", "fit_percent", "simulate_with_data",
]
