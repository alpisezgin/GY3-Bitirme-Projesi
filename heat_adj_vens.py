"""
Vensim .mdl → Python System Dynamics Simulator
Kaynak model: bitirme__heatadj_.mdl

Matematiksel Formüller (modelden otomatik çekilen):
────────────────────────────────────────────────────
STOCK 1: actual_temp
  d(actual_temp)/dt = (desired_temp - measured_temp) / adjustment_time

STOCK 2: merasured_temp
  d(measured_temp)/dt = (actual_temp - measured_temp) / measurement_delay

Başlangıç koşulları: actual_temp(0) = 0, measured_temp(0) = 0
Simülasyon: t = 0 → 100 s, dt = 0.1 s
"""

import re, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────
# 1. MDL PARSER
# ─────────────────────────────────────────────────────────────

def parse_mdl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = raw.split("\\\\\\---///")[0]
    stocks, constants, sim_controls = {}, {}, {}
    blocks = re.split(r'\|\s*\n', raw)

    for block in blocks:
        block = block.strip()
        if not block or block.startswith("*") or block.startswith("{"):
            continue

        m = re.match(r'^(.+?)\s*=\s*INTEG\s*\(\s*.+?,\s*([\-\d\.]+)\s*\)',
                     block, re.DOTALL | re.IGNORECASE)
        if m:
            stocks[m.group(1).strip().replace(" ", "_")] = float(m.group(2))
            continue

        first = block.split("\n")[0].strip()

        mc = re.match(r'^(FINAL TIME|INITIAL TIME|TIME STEP)\s*=\s*([\d\.]+)',
                      first, re.IGNORECASE)
        if mc:
            sim_controls[mc.group(1).upper()] = float(mc.group(2))
            continue

        me = re.match(r'^(.+?)\s*=\s*([\-\d\.]+)\s*$', first)
        if me:
            constants[me.group(1).strip().replace(" ", "_")] = float(me.group(2))

    return stocks, constants, sim_controls


# ─────────────────────────────────────────────────────────────
# 2. KULLANICIDAN 3 PARAMETRE AL
# ─────────────────────────────────────────────────────────────

def ask(prompt, default):
    while True:
        try:
            v = input(f"  {prompt} (varsayılan={default}): ").strip()
            return default if v == "" else float(v)
        except ValueError:
            print("    ⚠  Lütfen sayısal bir değer girin.")


def get_params(constants, sim_controls):
    print("\n" + "═" * 52)
    print("  HEAT ADJUSTMENT MODEL — Parametre Girişi")
    print("  Enter → varsayılan değeri kabul et")
    print("═" * 52 + "\n")

    desired_temp      = ask("desired_temp       [°C]  — hedef sıcaklık   ",
                             constants.get("desired_temp", 25.0))
    adjustment_time   = ask("adjustment_time    [s]   — ayarlama süresi  ",
                             constants.get("adjustment_time", 5.0))
    measurement_delay = ask("measurement_delay  [s]   — ölçüm gecikmesi  ",
                             constants.get("measurement_delay", 2.0))

    print("\n" + "═" * 52 + "\n")

    return {
        "desired_temp"      : desired_temp,
        "adjustment_time"   : adjustment_time,
        "measurement_delay" : measurement_delay,
        # Başlangıç koşulları ve simülasyon kontrolü modelden sabit
        "init_actual"       : 0.0,
        "init_measured"     : 0.0,
        "final_time"        : sim_controls.get("FINAL TIME", 300.0),
        "time_step"         : sim_controls.get("TIME STEP",  0.1),
    }


# ─────────────────────────────────────────────────────────────
# 3. EULER ENTEGRASYONU
# ─────────────────────────────────────────────────────────────

def run_sim(p):
    dt = p["time_step"]
    n  = int(round(p["final_time"] / dt))
    t  = np.linspace(0, p["final_time"], n + 1)

    actual   = np.zeros(n + 1)
    measured = np.zeros(n + 1)
    cit      = np.zeros(n + 1)
    cimt     = np.zeros(n + 1)

    actual[0]   = p["init_actual"]
    measured[0] = p["init_measured"]

    for i in range(n):
        cit[i]  = (p["desired_temp"] - measured[i]) / p["adjustment_time"]
        cimt[i] = (actual[i] - measured[i])          / p["measurement_delay"]
        actual[i+1]   = actual[i]   + cit[i]  * dt
        measured[i+1] = measured[i] + cimt[i] * dt

    cit[-1]  = (p["desired_temp"] - measured[-1]) / p["adjustment_time"]
    cimt[-1] = (actual[-1] - measured[-1])          / p["measurement_delay"]

    return {"time": t, "actual": actual, "measured": measured,
            "cit": cit, "cimt": cimt}


# ─────────────────────────────────────────────────────────────
# 4. GRAFİK
# ─────────────────────────────────────────────────────────────

def plot(res, p):
    t        = res["time"]
    actual   = res["actual"]
    measured = res["measured"]
    cit      = res["cit"]
    cimt     = res["cimt"]
    target   = p["desired_temp"]

    # Settle time (±2% bant)
    band = max(abs(target) * 0.02, 0.5)
    si   = np.where(np.abs(actual - target) <= band)[0]
    settle_t = t[si[0]] if len(si) > 0 else None

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 9), facecolor="#0b0f1a")
    fig.suptitle("Heat Adjustment Model — Sistem Davranışı",
                 fontsize=14, fontweight="bold", color="#e0e8f8",
                 y=0.97, fontfamily="monospace")

    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.42, wspace=0.30,
                            top=0.91, bottom=0.10, left=0.08, right=0.97)
    ax1 = fig.add_subplot(gs[0, :])   # üst — geniş: stocks
    ax2 = fig.add_subplot(gs[1, 0])   # alt-sol: flows
    ax3 = fig.add_subplot(gs[1, 1])   # alt-sağ: faz uzayı

    BG = "#10162a"
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#8090b0", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#2a3450")
        ax.grid(True, alpha=0.12, lw=0.6)

    # ── Panel 1: Stocks ─────────────────────────────────────
    ax1.plot(t, actual,   color="#e85d4a", lw=2.3, label="actual_temp",    zorder=3)
    ax1.plot(t, measured, color="#4a9eff", lw=2.3, label="merasured_temp",
             zorder=3, ls="--")
    ax1.axhline(target, color="#e85d4a", lw=1.0, ls=":", alpha=0.35)
    ax1.text(t[-1] * 0.98,
             target + (actual.max() - actual.min()) * 0.04,
             f"desired_temp = {target} °C",
             color="#e85d4a", fontsize=8, ha="right", alpha=0.75,
             fontfamily="monospace")

    if settle_t is not None:
        ax1.axvline(settle_t, color="#52c478", lw=1.0, ls="--", alpha=0.55)
        ax1.text(settle_t + t[-1] * 0.005,
                 actual.min() + (actual.max() - actual.min()) * 0.06,
                 f"settle ≈ {settle_t:.1f} s",
                 color="#52c478", fontsize=8, fontfamily="monospace")

    ax1.set_title("Stocks — Sıcaklık (°C)", color="#8090b0",
                  fontsize=10, pad=6, fontfamily="monospace")
    ax1.set_xlabel("Zaman (s)", color="#8090b0", fontsize=9)
    ax1.set_ylabel("°C", color="#8090b0", fontsize=9)
    ax1.legend(framealpha=0.15, fontsize=9, loc="lower right")

    # ── Panel 2: Flows ──────────────────────────────────────
    ax2.plot(t, cit,  color="#f0a500", lw=1.8, label="change_in_temperature")
    ax2.plot(t, cimt, color="#52c478", lw=1.8, label="change_in_measured_temp",
             ls="--")
    ax2.axhline(0, color="#ffffff", lw=0.5, alpha=0.2)
    ax2.set_title("Flows — Değişim Hızları", color="#8090b0",
                  fontsize=10, pad=6, fontfamily="monospace")
    ax2.set_xlabel("Zaman (s)", color="#8090b0", fontsize=9)
    ax2.set_ylabel("°C / s", color="#8090b0", fontsize=9)
    ax2.legend(framealpha=0.15, fontsize=8)

    # ── Panel 3: Faz Uzayı ──────────────────────────────────
    sc = ax3.scatter(actual, measured, c=t, cmap="plasma", s=3, alpha=0.8)
    ax3.plot(actual[0],  measured[0],  "o", color="#52c478",
             ms=7, label="Başlangıç", zorder=4)
    ax3.plot(actual[-1], measured[-1], "s", color="#e85d4a",
             ms=7, label="Son nokta", zorder=4)
    ax3.plot(target, target, "*", color="#ffffff", ms=10,
             label=f"Denge ({target},{target})", zorder=5)
    cb = plt.colorbar(sc, ax=ax3, pad=0.02)
    cb.ax.tick_params(colors="#8090b0", labelsize=8)
    cb.set_label("Zaman (s)", color="#8090b0", fontsize=8)
    ax3.set_title("Faz Uzayı: actual vs measured", color="#8090b0",
                  fontsize=10, pad=6, fontfamily="monospace")
    ax3.set_xlabel("actual_temp (°C)", color="#8090b0", fontsize=9)
    ax3.set_ylabel("merasured_temp (°C)", color="#8090b0", fontsize=9)
    ax3.legend(framealpha=0.15, fontsize=8)

    # ── Alt bilgi ───────────────────────────────────────────
    info = (f"desired_temp={p['desired_temp']} °C   |   "
            f"adjustment_time={p['adjustment_time']} s   |   "
            f"measurement_delay={p['measurement_delay']} s   |   "
            f"dt={p['time_step']} s   T={p['final_time']} s")
    fig.text(0.5, 0.015, info, ha="center", fontsize=8.5,
             color="#5a6a80", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#10162a",
                       edgecolor="#2a3450", alpha=0.85))

    plt.savefig("heat_model_output.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  ✅ Grafik kaydedildi: heat_model_output.png\n")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    mdl_path = sys.argv[1] if len(sys.argv) > 1 else \
               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bitirme__heatadj_.mdl")

    print(f"\n  📂 Model okunuyor: {mdl_path}")
    try:
        stocks, constants, sim_controls = parse_mdl(mdl_path)
    except FileNotFoundError:
        print("  ⚠  .mdl dosyası bulunamadı — varsayılan değerler kullanılacak.")
        stocks, constants, sim_controls = {}, {}, {}

    # Çekilen formülleri göster
    print("\n  ── Modelden Çekilen Matematiksel Formüller ──")
    print("  d(actual_temp)/dt    = (desired_temp - measured_temp) / adjustment_time")
    print("  d(measured_temp)/dt  = (actual_temp  - measured_temp) / measurement_delay")
    print(f"  Başlangıç koşulları  → actual_temp(0) = {stocks.get('actual_temp', 0)},"
          f"  measured_temp(0) = {stocks.get('merasured_temp', 0)}")
    print(f"  Simülasyon           → T = {sim_controls.get('FINAL TIME', 100)} s,"
          f"  dt = {sim_controls.get('TIME STEP', 0.1)} s")

    # Kullanıcıdan yalnızca 3 parametre al
    params = get_params(constants, sim_controls)

    # Simülasyon
    print("  🔄 Simülasyon çalışıyor...")
    results = run_sim(params)
    print("  ✅ Tamamlandı.\n")

    # Özet
    target = params["desired_temp"]
    actual = results["actual"]
    band   = max(abs(target) * 0.02, 0.5)
    si     = np.where(np.abs(actual - target) <= band)[0]
    print("  ── Sonuçlar ──")
    print(f"    final actual_temp    : {actual[-1]:.4f} °C")
    print(f"    final merasured_temp : {results['measured'][-1]:.4f} °C")
    if len(si) > 0:
        print(f"    settle time (±2%)   : {results['time'][si[0]]:.2f} s")
    else:
        print(f"    settle time (±2%)   : hedefe ulaşılamadı")

    # Grafik
    plot(results, params)


if __name__ == "__main__":
    main()