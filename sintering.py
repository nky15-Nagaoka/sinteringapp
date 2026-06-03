# sintering.app
# AI焼結シミュレーター / Web app prototype
# Streamlit web application with: dynamic sintering model selection, microstructure schematic,
# experiment feedback loop, and simple microstructure-property prediction.

import io
import json
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import differential_evolution

R = 8.314462618
EPS = 1e-30


# ============================================================
# Data classes
# ============================================================

@dataclass
class ThermalParams:
    T0_C: float = 25.0
    heating_rate_C_min: float = 10.0
    hold_T_C: float = 1400.0
    total_time_s: float = 3600.0
    dt_s: float = 2.0


@dataclass
class MicrostructureParams:
    rho0: float = 0.55
    G0_um: float = 0.50
    pore0: float = 0.45
    neck0: float = 0.05
    aspect_ratio: float = 1.0
    agglomeration_factor: float = 1.5
    open_closed_density: float = 0.92


@dataclass
class DiffusionParams:
    Ds0: float = 1e-8
    Db0: float = 1e-10
    Dv0: float = 1e-12
    Qs: float = 120e3
    Qb: float = 180e3
    Qv: float = 250e3
    gamma_s: float = 1.5
    gamma_b: float = 1.0
    surface_to_gb_switch_C: float = 900.0
    gb_to_lattice_switch_C: float = 1400.0


@dataclass
class AtmosphereFieldParams:
    pO2_atm: float = 0.21
    pH2O_atm: float = 1e-4
    E_V_m: float = 0.0
    J_A_m2: float = 0.0
    sample_length_mm: float = 5.0
    temp_gradient_C_mm: float = 0.0
    thermal_conductivity_W_mK: float = 10.0


@dataclass
class SecondPhaseParams:
    # Nanocomposite / second phase effects
    second_phase_vf: float = 0.03
    second_phase_radius_um: float = 0.20
    second_phase_hardness_GPa: float = 20.0
    matrix_hardness_GPa: float = 12.0
    interface_diffusion_boost: float = 1.0
    zener_strength: float = 1.0


@dataclass
class SinteringAidParams:
    # Sintering aid / glassy or liquid phase effects
    aid_wt_percent: float = 0.0
    liquidus_C: float = 1700.0
    eutectic_C: float = 1600.0
    wetting_factor: float = 0.5
    viscosity0_Pa_s: float = 1e6
    viscosity_Q_J_mol: float = 120e3
    aid_diffusion_boost: float = 1.0


@dataclass
class NoiseParams:
    impurity_offset_logD: float = 0.0
    Q_noise_sigma: float = 0.0
    random_seed: int = 42


@dataclass
class PropertyParams:
    H0_GPa: float = 12.0
    hall_petch_k: float = 0.45
    inverse_hp_critical_nm: float = 40.0
    inverse_hp_softening: float = 0.35
    porosity_model: str = "Rice"
    porosity_b: float = 4.0
    E_dense_GPa: float = 300.0
    KIC_dense_MPam05: float = 4.0
    flaw_size_um: float = 10.0
    indentation_h_um: float = 5.0


# ============================================================
# Physics core
# ============================================================

def temperature_profile(t, th: ThermalParams):
    rate_C_s = th.heating_rate_C_min / 60.0
    return min(th.T0_C + rate_C_s * t, th.hold_T_C) + 273.15


def joule_delta_T(atm: AtmosphereFieldParams):
    L = atm.sample_length_mm * 1e-3
    q = abs(atm.E_V_m * atm.J_A_m2)
    k = max(atm.thermal_conductivity_W_mK, 1e-6)
    return q * L * L / k


def defect_factors(T_K, atm: AtmosphereFieldParams):
    # qualitative defect chemistry / space-charge effect
    pO2 = max(atm.pO2_atm, 1e-30)
    pH2O = max(atm.pH2O_atm, 1e-30)
    oxygen_vacancy = np.clip(pO2 ** (-1.0 / 6.0), 0.05, 100.0)
    water_gb = np.clip(1.0 + 0.12 * np.log10(1 + pH2O * 1e6), 0.2, 10.0)
    field = np.clip(1.0 + 1e-8 * abs(atm.E_V_m) + 1e-10 * abs(atm.J_A_m2), 1.0, 100.0)
    return oxygen_vacancy, water_gb, field


def diffusion_coefficients(T_K, diff: DiffusionParams, atm: AtmosphereFieldParams,
                           sp: SecondPhaseParams, aid: SinteringAidParams, noise: NoiseParams):
    rng = np.random.default_rng(noise.random_seed)
    Qs = diff.Qs + rng.normal(0, noise.Q_noise_sigma)
    Qb = diff.Qb + rng.normal(0, noise.Q_noise_sigma)
    Qv = diff.Qv + rng.normal(0, noise.Q_noise_sigma)

    offset = 10 ** noise.impurity_offset_logD
    f_vac, f_h2o, f_field = defect_factors(T_K, atm)

    # Nanocomposite interfaces and sintering aid can enhance GB/interface transport.
    interface_boost = 1.0 + sp.second_phase_vf * sp.interface_diffusion_boost * 5.0
    aid_boost = 1.0 + aid.aid_wt_percent * 0.05 * aid.aid_diffusion_boost

    Ds = diff.Ds0 * offset * f_field * np.exp(-Qs / (R * T_K))
    Db = diff.Db0 * offset * f_h2o * f_field * interface_boost * aid_boost * np.exp(-Qb / (R * T_K))
    Dv = diff.Dv0 * offset * f_vac * f_field * np.exp(-Qv / (R * T_K))
    return Ds, Db, Dv


def liquid_viscosity(T_K, aid: SinteringAidParams):
    return aid.viscosity0_Pa_s * np.exp(aid.viscosity_Q_J_mol / (R * T_K))


def mechanism_flags(T_K, rho, G_um, diff: DiffusionParams, micro: MicrostructureParams, aid: SinteringAidParams):
    T_C = T_K - 273.15
    initial = rho < 0.75
    late = rho >= micro.open_closed_density
    surface = int(T_C < diff.surface_to_gb_switch_C and initial)
    gb = int((0.75 <= rho < micro.open_closed_density) or (diff.surface_to_gb_switch_C <= T_C < diff.gb_to_lattice_switch_C))
    lattice = int(T_C >= diff.gb_to_lattice_switch_C and late)
    liquid = int(T_C >= min(aid.liquidus_C, aid.eutectic_C) and aid.aid_wt_percent > 0)
    closed = int(rho >= micro.open_closed_density)
    abnormal = int(G_um > 5.0 * micro.G0_um * micro.agglomeration_factor)
    if abnormal:
        closed = 1
    return dict(surface_flag=surface, gb_flag=gb, lattice_flag=lattice, liquid_flag=liquid,
                closed_pore_flag=closed, abnormal_grain_growth_flag=abnormal)


def zener_pinning(sp: SecondPhaseParams, pore, G_um):
    # Fpin increases with fine second phase particles and pores.
    r = max(sp.second_phase_radius_um, 1e-6)
    F_zener = sp.zener_strength * 3.0 * sp.second_phase_vf / r
    F_pore = 2.0 * pore / max(G_um, 1e-6)
    return F_zener + F_pore


def grain_growth_rate(T_K, G_um, rho, pore, diff: DiffusionParams, sp: SecondPhaseParams, flags):
    Mb0 = 1e-8
    Qm = 180e3
    Mb = Mb0 * np.exp(-Qm / (R * T_K))
    G_m = max(G_um * 1e-6, 1e-12)
    Fpin = zener_pinning(sp, pore, G_um)
    liquid_multiplier = 2.0 if flags["liquid_flag"] else 1.0
    driving = max(2 * diff.gamma_b / G_m - Fpin, 0.0)
    return liquid_multiplier * Mb * driving * 1e6


def neck_growth_rate(Ds, Db, Dv, G_um, flags):
    G_m = max(G_um * 1e-6, 1e-12)
    # surface diffusion: strong neck growth but weak densification
    contribution = 2.0 * Ds + 0.7 * Db + 0.3 * Dv
    if flags["liquid_flag"]:
        contribution *= 5.0
    return 1e-3 * contribution / max(G_m ** 4, EPS)


def densification_rate(model, T_K, rho, G_um, pore, neck, Ds, Db, Dv,
                       diff: DiffusionParams, micro: MicrostructureParams,
                       sp: SecondPhaseParams, aid: SinteringAidParams, flags):
    G_m = max(G_um * 1e-6, 1e-12)
    gamma = max(diff.gamma_b, 1e-6)
    geom = 1.0 / max(micro.aspect_ratio, 0.1)
    agglom = 1.0 / max(micro.agglomeration_factor, 1.0)
    pore_drive = max(1.0 - rho, 0.0)
    closed_penalty = 0.25 if flags["closed_pore_flag"] else 1.0
    zener_penalty = 1.0 / (1.0 + 0.1 * zener_pinning(sp, pore, G_um))

    # Surface diffusion competition suppresses densification.
    surface_fraction = Ds / max(Ds + Db + Dv, EPS)
    surface_penalty = max(0.05, 1.0 - 0.85 * surface_fraction)

    if model == "Coble（粒界拡散支配）":
        # Coble creep style: GB diffusion, strong grain-size dependence.
        rate = 5e5 * gamma * Db / max(G_m ** 3, EPS)
    elif model == "Nabarro-Herring（格子拡散支配）":
        # Lattice diffusion, weaker grain-size dependence.
        rate = 2e2 * gamma * Dv / max(G_m ** 2, EPS)
    elif model == "Kingery型液相焼結":
        # Rearrangement + solution-precipitation, controlled by liquid amount, wetting, viscosity.
        eta = max(liquid_viscosity(T_K, aid), 1e-9)
        liquid_fraction = np.clip(0.01 * aid.aid_wt_percent, 0.0, 0.35)
        capillary = diff.gamma_s / max(G_m, 1e-12)
        rearrangement = 1e-5 * liquid_fraction * aid.wetting_factor * capillary / eta
        solution_precip = 1e6 * liquid_fraction * aid.wetting_factor * Db / max(G_m ** 2, EPS)
        rate = rearrangement + solution_precip
        if not flags["liquid_flag"]:
            rate *= 0.05
    else:  # automatic hybrid
        w_coble = 1.0 if flags["gb_flag"] else 0.2
        w_nh = 1.0 if flags["lattice_flag"] else 0.2
        w_liq = 1.0 if flags["liquid_flag"] else 0.0
        rate_c = 5e5 * gamma * Db / max(G_m ** 3, EPS)
        rate_n = 2e2 * gamma * Dv / max(G_m ** 2, EPS)
        eta = max(liquid_viscosity(T_K, aid), 1e-9)
        lf = np.clip(0.01 * aid.aid_wt_percent, 0.0, 0.35)
        rate_l = 1e6 * lf * aid.wetting_factor * Db / max(G_m ** 2, EPS) + 1e-5 * lf * diff.gamma_s / max(G_m, 1e-12) / eta
        rate = (w_coble * rate_c + w_nh * rate_n + w_liq * rate_l) / max(w_coble + w_nh + w_liq, 1e-6)

    return 1e-4 * rate * pore_drive * geom * agglom * closed_penalty * zener_penalty * surface_penalty


def simulate(model, th, micro, diff, atm, sp, aid, noise):
    steps = int(th.total_time_s // th.dt_s) + 1
    rho = micro.rho0
    G = micro.G0_um
    pore = max(micro.pore0, 1.0 - rho)
    neck = micro.neck0
    rows = []

    for i in range(steps):
        t = i * th.dt_s
        T_K = temperature_profile(t, th)
        T_K += atm.temp_gradient_C_mm * atm.sample_length_mm / 2.0
        jdT = joule_delta_T(atm)
        T_K += jdT
        runaway = int(jdT > 100)

        Ds, Db, Dv = diffusion_coefficients(T_K, diff, atm, sp, aid, noise)
        flags = mechanism_flags(T_K, rho, G, diff, micro, aid)
        flags["thermal_runaway_flag"] = runaway
        Deff = (0.1 + flags["surface_flag"]) * Ds + (0.1 + flags["gb_flag"]) * Db + (0.05 + flags["lattice_flag"]) * Dv
        if flags["liquid_flag"]:
            Deff *= 20

        drho = densification_rate(model, T_K, rho, G, pore, neck, Ds, Db, Dv, diff, micro, sp, aid, flags)
        dG = grain_growth_rate(T_K, G, rho, pore, diff, sp, flags)
        dX = neck_growth_rate(Ds, Db, Dv, G, flags)

        rho = float(np.clip(rho + drho * th.dt_s, micro.rho0, 0.999))
        G = float(max(G + dG * th.dt_s, 1e-6))
        pore = float(max(1.0 - rho, 0.0))
        neck = float(np.clip(neck + dX * th.dt_s, 0.0, 1.0))

        rows.append({
            "t": t, "T_C": T_K - 273.15, "rho": rho, "G_um": G, "pore": pore, "neck_ratio": neck,
            "Ds": Ds, "Db": Db, "Dv": Dv, "Deff": Deff, "d_rho_dt": drho, "dG_dt": dG,
            **flags
        })
    return pd.DataFrame(rows)


# ============================================================
# Property prediction
# ============================================================

def porosity_factor(P, model, b):
    P = np.clip(P, 0.0, 0.8)
    if model == "Rice":
        return np.exp(-b * P)
    if model == "Knudsen":
        return max(0.0, 1 - P) ** b
    if model == "Schiller":
        return max(0.0, 1 - 1.9 * P + 0.9 * P ** 2)
    if model == "Hasselman":
        return max(0.0, 1 - P) / max(1 + b * P, 1e-9)
    return np.exp(-b * P)


def predict_properties(df, prop: PropertyParams, sp: SecondPhaseParams):
    out = df.copy()
    G_um = np.maximum(out["G_um"].to_numpy(), 1e-6)
    G_nm = G_um * 1000.0
    P = out["pore"].to_numpy()

    # Hall-Petch with simple inverse softening below critical nanograin size.
    H_hp = prop.H0_GPa + prop.hall_petch_k / np.sqrt(G_um)
    soft = np.ones_like(G_um)
    mask = G_nm < prop.inverse_hp_critical_nm
    soft[mask] = 1.0 - prop.inverse_hp_softening * (1.0 - G_nm[mask] / prop.inverse_hp_critical_nm)
    H_matrix = H_hp * np.clip(soft, 0.1, 1.0)

    fP = np.array([porosity_factor(x, prop.porosity_model, prop.porosity_b) for x in P])
    H_dense = H_matrix * fP

    # Voigt / Reuss / Hill hardness estimate for composite effect.
    vf = np.clip(sp.second_phase_vf, 0.0, 0.8)
    H2 = sp.second_phase_hardness_GPa
    H1 = H_dense
    H_voigt = (1 - vf) * H1 + vf * H2
    H_reuss = 1.0 / np.maximum((1 - vf) / np.maximum(H1, 1e-6) + vf / max(H2, 1e-6), 1e-9)
    H_hill = 0.5 * (H_voigt + H_reuss)

    E = prop.E_dense_GPa * fP
    # Griffith-like: lower porosity and smaller defects improve fracture-related estimate.
    a_m = max(prop.flaw_size_um * 1e-6, 1e-12)
    gamma_fracture = 5.0  # J/m2, pedagogical default
    sigma_griffith_MPa = np.sqrt(2 * (E * 1e9) * gamma_fracture / (np.pi * a_m)) / 1e6
    KIC = prop.KIC_dense_MPam05 * np.sqrt(np.clip(1 - P, 0, 1))

    # Nix-Gao indentation size effect: H^2 = H0^2(1+h*/h)
    hstar_um = 0.5
    H_nix_gao = H_hill * np.sqrt(1.0 + hstar_um / max(prop.indentation_h_um, 1e-6))

    out["hardness_HallPetch_porosity_GPa"] = H_dense
    out["hardness_Voigt_GPa"] = H_voigt
    out["hardness_Reuss_GPa"] = H_reuss
    out["hardness_Hill_GPa"] = H_hill
    out["hardness_NixGao_GPa"] = H_nix_gao
    out["elastic_modulus_GPa"] = E
    out["griffith_strength_MPa"] = sigma_griffith_MPa
    out["fracture_toughness_est_MPam05"] = KIC
    return out


# ============================================================
# Microstructure schematic
# ============================================================

def draw_microstructure(rho, G_um, pore, second_vf, aid_wt, neck, seed=1):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_grains = int(np.clip(18 / np.sqrt(max(G_um, 0.03)), 8, 80))
    base_r = np.clip(0.045 * np.sqrt(G_um / 0.5), 0.018, 0.09)
    shrink = 0.75 + 0.45 * rho

    for _ in range(n_grains):
        x, y = rng.uniform(0.08, 0.92, 2)
        r = base_r * shrink * rng.uniform(0.75, 1.25)
        c = Circle((x, y), r, fill=False, linewidth=1.1, alpha=0.75)
        ax.add_patch(c)
        # neck indication
        if rng.random() < neck:
            ax.plot([x, np.clip(x + rng.normal(0, 0.08), 0.05, 0.95)],
                    [y, np.clip(y + rng.normal(0, 0.08), 0.05, 0.95)], linewidth=0.7, alpha=0.35)

    n_pores = int(np.clip(80 * pore, 2, 45))
    for _ in range(n_pores):
        x, y = rng.uniform(0.06, 0.94, 2)
        r = rng.uniform(0.005, 0.025) * (1.3 - rho)
        ax.add_patch(Circle((x, y), r, color="black", alpha=0.55))

    n_second = int(np.clip(250 * second_vf, 0, 70))
    for _ in range(n_second):
        x, y = rng.uniform(0.06, 0.94, 2)
        r = rng.uniform(0.003, 0.008)
        ax.add_patch(Circle((x, y), r, color="gray", alpha=0.85))

    # glass/liquid phase schematic: thin translucent network
    if aid_wt > 0:
        for _ in range(int(np.clip(3 * aid_wt, 1, 40))):
            x = np.linspace(0.05, 0.95, 60)
            y0 = rng.uniform(0.05, 0.95)
            y = y0 + 0.015 * np.sin(10 * x + rng.uniform(0, 6.28))
            ax.plot(x, y, alpha=0.12, linewidth=2)

    ax.text(0.02, 0.98, f"ρ={rho:.3f}\nG={G_um:.3f} µm\nP={pore:.3f}", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    return fig


# ============================================================
# Feedback / inverse fitting
# ============================================================

def fit_from_experiment(exp_df, model, th, micro, diff, atm, sp, aid, noise):
    required = {"t"}
    if not required.issubset(exp_df.columns):
        raise ValueError("CSVには少なくとも 't' 列が必要です。")
    has_rho = "rho_exp" in exp_df.columns
    has_G = "G_um_exp" in exp_df.columns
    if not (has_rho or has_G):
        raise ValueError("CSVには 'rho_exp' または 'G_um_exp' のどちらかが必要です。")

    t_exp = exp_df["t"].to_numpy()

    def loss(x):
        log_Db0, log_Dv0, Qb_scale, aid_boost, agglom, zener = x
        d2 = DiffusionParams(**asdict(diff))
        a2 = SinteringAidParams(**asdict(aid))
        m2 = MicrostructureParams(**asdict(micro))
        s2 = SecondPhaseParams(**asdict(sp))
        d2.Db0 = 10 ** log_Db0
        d2.Dv0 = 10 ** log_Dv0
        d2.Qb = diff.Qb * Qb_scale
        a2.aid_diffusion_boost = aid_boost
        m2.agglomeration_factor = agglom
        s2.zener_strength = zener
        sim = simulate(model, th, m2, d2, atm, s2, a2, noise)
        err = 0.0
        if has_rho:
            pred = np.interp(t_exp, sim["t"], sim["rho"])
            err += np.mean((pred - exp_df["rho_exp"].to_numpy()) ** 2)
        if has_G:
            pred = np.interp(t_exp, sim["t"], sim["G_um"])
            # normalized grain-size error
            denom = max(np.nanmean(exp_df["G_um_exp"]), 1e-6)
            err += np.mean(((pred - exp_df["G_um_exp"].to_numpy()) / denom) ** 2)
        return float(err)

    bounds = [(-16, -6), (-16, -6), (0.7, 1.3), (0.2, 10.0), (1.0, 5.0), (0.2, 5.0)]
    res = differential_evolution(loss, bounds, maxiter=25, popsize=8, polish=False, seed=noise.random_seed)
    keys = ["log10_Db0", "log10_Dv0", "Qb_scale", "aid_diffusion_boost", "agglomeration_factor", "zener_strength"]
    return dict(zip(keys, res.x)), res.fun


def experiment_guidance():
    st.markdown("""
### 実験フィードバックの入れ方
精度向上には、まず次の3種類の実験値を入れるのが効果的です。

1. **TMA / Dilatometer**: 時間 `t` と相対密度 `rho_exp`、または収縮率から換算した密度。  
2. **SEM / EBSD**: 各焼結時間・温度での平均粒径 `G_um_exp`。  
3. **アルキメデス密度・画像解析気孔率**: 最終密度、開気孔率、閉気孔率。

CSV例:
```text
t,rho_exp,G_um_exp
0,0.55,0.50
600,0.61,0.53
1200,0.73,0.61
1800,0.86,0.80
3600,0.95,1.30
```

このアプリでは、実験CSVを入れるとまず以下を補正します。

- `Db0`, `Dv0`: 粒界拡散・格子拡散の前因子
- `Qb`: 粒界拡散の活性化エネルギー補正
- 焼結助剤の拡散促進係数
- 凝集係数
- 第二相/Zenerピン止め強度

初学者向けのおすすめ手順は、**同じ粉末で焼結温度だけを3水準、保持時間を3水準**にした9条件を測り、TMA曲線とSEM粒径を同じCSV形式で追加することです。
""")


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="AI焼結シミュレーター", layout="wide")
st.title("AI焼結シミュレーター")
st.caption("拡散機構切替・ナノコンポジット/第二相・焼結助剤・実験フィードバック・微細構造-物性予測を一体化したWebアプリ試作版")

with st.sidebar:
    st.header("音楽イコライザー風 入力スライダー")
    model = st.selectbox("焼結モデル", ["自動ハイブリッド", "Coble（粒界拡散支配）", "Nabarro-Herring（格子拡散支配）", "Kingery型液相焼結"])

    st.subheader("温度・時間")
    th = ThermalParams(
        T0_C=st.number_input("初期温度 [°C]", value=25.0),
        heating_rate_C_min=st.slider("昇温速度 [°C/min]", 0.1, 50.0, 10.0),
        hold_T_C=st.slider("保持温度 [°C]", 300.0, 2200.0, 1400.0),
        total_time_s=float(st.slider("総時間 [s]", 60, 20000, 3600)),
        dt_s=st.slider("時間刻み dt [s]", 0.5, 20.0, 2.0),
    )

    st.subheader("微細構造")
    micro = MicrostructureParams(
        rho0=st.slider("初期相対密度", 0.30, 0.80, 0.55),
        G0_um=st.slider("初期粒径 [µm]", 0.01, 20.0, 0.50),
        pore0=st.slider("初期気孔率", 0.05, 0.70, 0.45),
        neck0=st.slider("初期ネック比", 0.00, 0.50, 0.05),
        aspect_ratio=st.slider("粒子アスペクト比", 0.2, 5.0, 1.0),
        agglomeration_factor=st.slider("凝集係数", 1.0, 5.0, 1.5),
        open_closed_density=st.slider("閉気孔化密度", 0.85, 0.98, 0.92),
    )

    st.subheader("拡散・エネルギー")
    diff = DiffusionParams(
        Ds0=st.number_input("Ds0 [m²/s]", value=1e-8, format="%.3e"),
        Db0=st.number_input("Db0 [m²/s]", value=1e-10, format="%.3e"),
        Dv0=st.number_input("Dv0 [m²/s]", value=1e-12, format="%.3e"),
        Qs=st.number_input("Qs [J/mol]", value=120e3, format="%.3e"),
        Qb=st.number_input("Qb [J/mol]", value=180e3, format="%.3e"),
        Qv=st.number_input("Qv [J/mol]", value=250e3, format="%.3e"),
        gamma_s=st.slider("表面エネルギー γs [J/m²]", 0.1, 3.0, 1.5),
        gamma_b=st.slider("粒界エネルギー γb [J/m²]", 0.1, 3.0, 1.0),
        surface_to_gb_switch_C=st.slider("表面→粒界拡散切替 [°C]", 300.0, 1600.0, 900.0),
        gb_to_lattice_switch_C=st.slider("粒界→格子拡散切替 [°C]", 800.0, 2200.0, 1400.0),
    )

    st.subheader("雰囲気・電場")
    atm = AtmosphereFieldParams(
        pO2_atm=st.number_input("pO2 [atm]", value=0.21, format="%.3e"),
        pH2O_atm=st.number_input("pH2O [atm]", value=1e-4, format="%.3e"),
        E_V_m=st.number_input("電場 E [V/m]", value=0.0, format="%.3e"),
        J_A_m2=st.number_input("電流密度 J [A/m²]", value=0.0, format="%.3e"),
        sample_length_mm=st.slider("試料長さ [mm]", 0.1, 50.0, 5.0),
        temp_gradient_C_mm=st.slider("温度勾配 [°C/mm]", 0.0, 50.0, 0.0),
        thermal_conductivity_W_mK=st.slider("熱伝導率 [W/mK]", 0.1, 200.0, 10.0),
    )

    st.subheader("第二相・ナノコンポジット")
    sp = SecondPhaseParams(
        second_phase_vf=st.slider("第二相体積分率", 0.0, 0.50, 0.03),
        second_phase_radius_um=st.slider("第二相粒径 [µm]", 0.005, 5.0, 0.20),
        second_phase_hardness_GPa=st.slider("第二相硬度 [GPa]", 1.0, 60.0, 20.0),
        matrix_hardness_GPa=st.slider("母相硬度 [GPa]", 1.0, 40.0, 12.0),
        interface_diffusion_boost=st.slider("界面拡散促進", 0.1, 10.0, 1.0),
        zener_strength=st.slider("Zenerピン止め強度", 0.1, 10.0, 1.0),
    )

    st.subheader("焼結助剤・液相")
    aid = SinteringAidParams(
        aid_wt_percent=st.slider("焼結助剤 [wt%]", 0.0, 20.0, 0.0),
        liquidus_C=st.slider("液相生成温度 [°C]", 500.0, 2200.0, 1700.0),
        eutectic_C=st.slider("共晶温度 [°C]", 500.0, 2200.0, 1600.0),
        wetting_factor=st.slider("濡れ性係数", 0.0, 1.0, 0.5),
        viscosity0_Pa_s=st.number_input("液相粘度係数 η0 [Pa s]", value=1e6, format="%.3e"),
        viscosity_Q_J_mol=st.number_input("粘度活性化項 [J/mol]", value=120e3, format="%.3e"),
        aid_diffusion_boost=st.slider("助剤による拡散促進", 0.1, 10.0, 1.0),
    )

    st.subheader("物性予測")
    prop = PropertyParams(
        H0_GPa=st.slider("基準硬度 H0 [GPa]", 1.0, 40.0, 12.0),
        hall_petch_k=st.slider("Hall-Petch係数 k", 0.01, 2.0, 0.45),
        inverse_hp_critical_nm=st.slider("逆Hall-Petch開始粒径 [nm]", 5.0, 200.0, 40.0),
        inverse_hp_softening=st.slider("逆Hall-Petch軟化係数", 0.0, 0.9, 0.35),
        porosity_model=st.selectbox("気孔率補正", ["Rice", "Knudsen", "Schiller", "Hasselman"]),
        porosity_b=st.slider("気孔率モデル係数", 0.1, 10.0, 4.0),
        E_dense_GPa=st.slider("緻密体ヤング率 [GPa]", 10.0, 600.0, 300.0),
        KIC_dense_MPam05=st.slider("緻密体KIC [MPa√m]", 0.5, 20.0, 4.0),
        flaw_size_um=st.slider("代表欠陥サイズ [µm]", 0.1, 200.0, 10.0),
        indentation_h_um=st.slider("圧痕深さ h [µm]", 0.05, 50.0, 5.0),
    )

    noise = NoiseParams(
        impurity_offset_logD=st.slider("不純物logDオフセット", -3.0, 3.0, 0.0),
        Q_noise_sigma=st.slider("Qノイズ σ [J/mol]", 0.0, 50000.0, 0.0),
        random_seed=int(st.number_input("乱数シード", value=42, step=1)),
    )


# Run simulation continuously so sliders feel interactive.
df = simulate(model, th, micro, diff, atm, sp, aid, noise)
dfp = predict_properties(df, prop, sp)

# Tabs keep a single integrated user experience.
tab1, tab2, tab3, tab4, tab5 = st.tabs(["微細構造ビュー", "時系列", "物性予測", "実験フィードバック", "入出力"])

with tab1:
    st.subheader("スライダーで変わる内部構造の模式図")
    st.write("左のイコライザー風スライダーを動かすと、密度・粒径・気孔率・第二相・焼結助剤の効果が模式図に反映されます。")
    time_pick = st.slider("観察時刻 [s]", 0.0, float(df["t"].max()), float(df["t"].max()), step=float(th.dt_s))
    row = df.iloc[(df["t"] - time_pick).abs().argmin()]
    colA, colB = st.columns([1, 1])
    with colA:
        fig = draw_microstructure(row["rho"], row["G_um"], row["pore"], sp.second_phase_vf,
                                  aid.aid_wt_percent, row["neck_ratio"], seed=noise.random_seed + int(row["t"]))
        st.pyplot(fig)
    with colB:
        st.metric("相対密度 ρ", f"{row['rho']:.3f}")
        st.metric("平均粒径 G", f"{row['G_um']:.3f} µm")
        st.metric("気孔率 P", f"{row['pore']:.3f}")
        st.metric("ネック比 X/G", f"{row['neck_ratio']:.3f}")
        flags = [c for c in df.columns if c.endswith("flag")]
        st.dataframe(row[flags].astype(int).to_frame("flag"), use_container_width=True)

with tab2:
    st.subheader("密度・粒径・気孔率の時系列")
    st.line_chart(dfp.set_index("t")[["rho", "G_um", "pore", "neck_ratio"]])
    st.subheader("拡散係数")
    st.line_chart(dfp.set_index("t")[["Ds", "Db", "Dv", "Deff"]])
    st.subheader("機構切替フラグ")
    flag_cols = [c for c in dfp.columns if c.endswith("flag")]
    st.line_chart(dfp.set_index("t")[flag_cols])

with tab3:
    st.subheader("微細構造からの簡易物性予測")
    st.write("硬度はHall-Petch/逆Hall-Petch、気孔率補正、Voigt-Reuss-Hill混合則、Nix-Gao型圧痕サイズ効果を簡易的に組み合わせています。")
    st.line_chart(dfp.set_index("t")[["hardness_Hill_GPa", "hardness_NixGao_GPa", "elastic_modulus_GPa", "fracture_toughness_est_MPam05"]])
    st.dataframe(dfp.tail(10)[["t", "rho", "G_um", "pore", "hardness_Hill_GPa", "hardness_NixGao_GPa", "elastic_modulus_GPa", "griffith_strength_MPa", "fracture_toughness_est_MPam05"]], use_container_width=True)

with tab4:
    experiment_guidance()
    uploaded = st.file_uploader("実験CSVをアップロード", type=["csv"])
    if uploaded is not None:
        exp_df = pd.read_csv(uploaded)
        st.write("読み込んだ実験データ")
        st.dataframe(exp_df, use_container_width=True)
        if st.button("実験データでパラメタ補正"):
            try:
                fitted, loss = fit_from_experiment(exp_df, model, th, micro, diff, atm, sp, aid, noise)
                st.success(f"補正が完了しました。Loss={loss:.4e}")
                st.json(fitted)
                st.info("上の値を左サイドバーの対応パラメタへ反映すると、次回シミュレーションに使えます。将来版では自動反映・履歴保存に拡張できます。")
            except Exception as e:
                st.error(str(e))

with tab5:
    st.subheader("ダウンロード")
    st.download_button("シミュレーション結果CSV", dfp.to_csv(index=False).encode("utf-8"), "sintering_result.csv", "text/csv")
    config = {
        "model": model, "thermal": asdict(th), "microstructure": asdict(micro), "diffusion": asdict(diff),
        "atmosphere_field": asdict(atm), "second_phase": asdict(sp), "sintering_aid": asdict(aid),
        "noise": asdict(noise), "property": asdict(prop)
    }
    st.download_button("入力設定JSON", json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8"), "sintering_config.json", "application/json")
    st.subheader("注意")
    st.warning("本アプリは研究設計・教育・初期スクリーニング用の半定量モデルです。論文値や実測TMA/SEM/密度データで校正してから材料設計判断に使ってください。")
