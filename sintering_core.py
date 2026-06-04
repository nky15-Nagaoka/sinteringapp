# sintering_core.py
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

R = 8.314

@dataclass
class Params:
    mode: str = "Research"
    model: str = "自動ハイブリッド"
    total_time_s: float = 3600
    dt: float = 2
    T0_C: float = 25
    target_temp_C: float = 1400
    heating_rate_C_min: float = 10
    hold_time_s: float = 1800
    rho0: float = 0.55
    G0_um: float = 0.5
    Ds0: float = 1e-8
    Db0: float = 1e-10
    Dv0: float = 1e-12
    Qs: float = 120e3
    Qb: float = 180e3
    Qv: float = 260e3
    gamma_b: float = 1.0
    gamma_s: float = 1.5
    liquid_temp_C: float = 1600
    viscosity_log10_Pa_s_at_liquid: float = 3.0
    second_phase_fraction: float = 0.03
    second_phase_radius_um: float = 0.2
    sintering_aid_fraction: float = 0.02
    aid_effect_strength: float = 1.5
    pO2_atm: float = 0.21
    pH2O_atm: float = 1e-4
    E_V_m: float = 0
    J_A_m2: float = 0
    open_closed_density: float = 0.92
    aspect_ratio: float = 1.0
    agglomeration_factor: float = 1.2
    abnormal_G_um: float = 5.0
    E0_GPa: float = 300
    H0_GPa: float = 15
    KIC0_MPam05: float = 3
    hp_k: float = 0.15
    inverse_hp_threshold_um: float = 0.08

    # 試料サイズ・炉・熱伝導（Digital Twinモードで1D熱伝導＋焼結連成に使用）
    sample_shape: str = "円板"  # 円板 / 円柱 / 角板
    sample_diameter_mm: float = 10.0
    sample_thickness_mm: float = 3.0
    sample_width_mm: float = 10.0
    furnace_inner_diameter_mm: float = 80.0
    furnace_uniformity_C: float = 5.0
    thermal_conductivity_W_mK: float = 10.0
    density_bulk_kg_m3: float = 3900.0
    heat_capacity_J_kgK: float = 800.0
    heat_transfer_W_m2K: float = 80.0
    thermal_nodes: int = 11

    # UI/研究用途の半定量調整: 文献D0だけでは過小評価になりやすいため、
    # TMA等で同定する緻密化倍率と到達密度を明示的に持たせる。
    densification_scale: float = 50.0
    target_final_density: float = 0.995


def temp_profile(t, p: Params):
    ramp = p.heating_rate_C_min / 60.0
    return min(p.T0_C + ramp*t, p.target_temp_C) + 273.15


def defect_and_field_factors(T, p: Params):
    pO2 = max(p.pO2_atm, 1e-30)
    pH2O = max(p.pH2O_atm, 1e-30)
    f_o = np.clip(pO2 ** (-1/10), 0.2, 5.0)
    f_h = np.clip(1 + 0.08*np.log10(1 + pH2O*1e6), 0.5, 3.0)
    f_e = np.clip(1 + 3e-9*abs(p.E_V_m) + 5e-11*abs(p.J_A_m2), 1.0, 8.0)
    return f_o, f_h, f_e


def diffusivities(T, p: Params):
    f_o, f_h, f_e = defect_and_field_factors(T, p)
    aid = 1.0 + p.aid_effect_strength * p.sintering_aid_fraction * 10.0
    # MgOなど粒成長抑制助剤はaid_effect_strength<1で表現可能
    Ds = p.Ds0 * f_e * aid * np.exp(-p.Qs/(R*T))
    Db = p.Db0 * f_h * f_e * aid * np.exp(-p.Qb/(R*T))
    Dv = p.Dv0 * f_o * f_e * np.sqrt(aid) * np.exp(-p.Qv/(R*T))
    return Ds, Db, Dv


def flags(T, rho, G, p: Params):
    TC = T - 273.15
    liquid = int(TC >= p.liquid_temp_C)
    closed = int(rho >= p.open_closed_density)
    abnormal = int(G >= p.abnormal_G_um)
    thermal_runaway = int(abs(p.E_V_m*p.J_A_m2) > 5e8)
    surface = int(rho < 0.72 and not liquid)
    gb = int(0.62 <= rho < p.open_closed_density and not liquid)
    lattice = int(rho >= 0.85 and not liquid)
    if abnormal:
        closed = 1
    return dict(surface=surface, gb=gb, lattice=lattice, liquid=liquid, closed=closed,
                abnormal=abnormal, thermal_runaway=thermal_runaway)


def zener_pin(pore, G, p: Params):
    r = max(p.second_phase_radius_um, 1e-4)
    fp = 3*p.second_phase_fraction/(2*r)
    pore_pin = 1.5*pore/max(G, 1e-4)
    return fp + pore_pin


def densification_model(T, rho, G, pore, Ds, Db, Dv, fl, p: Params):
    Gm = max(G*1e-6, 1e-10)
    model = p.model
    if model == "自動ハイブリッド":
        model = "Kingery液相" if fl['liquid'] else ("Coble粒界拡散" if rho < 0.9 else "Nabarro-Herring格子拡散")

    # normalized semi-quantitative kinetic terms
    if model == "Coble粒界拡散":
        k = Db / Gm**3
        mechanism_factor = 1.0
    elif model == "Nabarro-Herring格子拡散":
        k = Dv / Gm**2
        mechanism_factor = 0.6
    elif model == "Kingery液相":
        eta = 10**p.viscosity_log10_Pa_s_at_liquid
        capillary = max(p.gamma_s, 0.1)/eta
        wetting = 1 + 20*p.sintering_aid_fraction*p.aid_effect_strength
        rearrangement = capillary*wetting/(Gm**1.5)
        solution_precip = (Db + Dv)/(Gm**2) * wetting
        k = 1e-8*rearrangement + solution_precip
        mechanism_factor = 3.0 if fl['liquid'] else 0.15
    else:
        k = Db/Gm**3
        mechanism_factor = 1.0

    geom = 1/max(p.aspect_ratio, 0.2)/max(p.agglomeration_factor, 1.0)
    closed_penalty = 0.25 if fl['closed'] else 1.0
    pin_penalty = 1/(1 + 0.8*zener_pin(pore, G, p))
    surface_loss = Ds/max(Ds+Db+Dv, 1e-300)

    # 旧版は drive=(1-rho)^1.2 のため、後期焼結で急停止しやすかった。
    # ここでは「目標到達密度」へ漸近する駆動力に変更し、
    # 完全緻密化に近い 0.99 以上まで計算できるようにする。
    rho_limit = float(np.clip(p.target_final_density, p.rho0 + 0.02, 0.999))
    residual_drive = max(rho_limit - rho, 0.0)
    drive = residual_drive**1.05

    # 半定量モデルの絶対速度をTMA/実験で合わせ込むための倍率。
    # プリセット値は文献値の初期シードなので、研究時はここを実験で校正する。
    scale = max(float(p.densification_scale), 1e-6)

    rate = 2e-4*k*drive*geom*closed_penalty*pin_penalty*mechanism_factor*scale
    rate *= max(0.05, 1-0.65*surface_loss)

    # Kingery液相や助剤系では粒子再配列・溶解析出で後期も進みやすいので、
    # 温度が十分高い場合のみ弱い経験的クロージャ項を加える。
    TC = T - 273.15
    if TC >= 0.92*p.target_temp_C:
        hold_strength = np.clip((TC - 0.92*p.target_temp_C)/max(0.08*p.target_temp_C, 1.0), 0, 1)
        aid_boost = 1.0 + 8.0*p.sintering_aid_fraction*p.aid_effect_strength
        late_boost = 3.0 if fl['liquid'] else 1.0
        empirical = 2.0e-5 * scale * hold_strength * aid_boost * late_boost * residual_drive
        empirical *= (0.35 if fl['closed'] else 1.0)
        rate += empirical

    # target_final_densityを超えて暴走しないようにsimulate側でもクリップする。
    return max(rate, 0), model


def grain_growth(T, rho, G, pore, p: Params, fl):
    Mb0 = 2e-8
    Qm = 0.75*p.Qb
    Mb = Mb0*np.exp(-Qm/(R*T))
    pin = zener_pin(pore, G, p)
    Gm = max(G*1e-6, 1e-10)
    driving = max(2*p.gamma_b/Gm - pin, 0)
    rate = Mb*driving*1e6
    if fl['liquid']:
        rate *= 2.5
    # nanocomposite/second phase pinning suppresses coarsening
    rate /= (1 + 30*p.second_phase_fraction/max(p.second_phase_radius_um, 0.03))
    return max(rate, 0)


def neck_rate(G, Ds, Db, Dv, fl):
    Gm = max(G*1e-6, 1e-10)
    k = (2*Ds + 0.5*Db + 0.1*Dv)/Gm**4
    if fl['liquid']:
        k *= 3
    return min(5e-3*k, 0.05)



def _geometry_half_length_m(p: Params):
    """1D熱伝導で解く代表長さ。円板/角板は厚み方向、円柱は半径方向を優先。"""
    shape = str(getattr(p, "sample_shape", "円板"))
    if "円柱" in shape:
        return max(float(p.sample_diameter_mm) * 1e-3 / 2.0, 1e-4)
    return max(float(p.sample_thickness_mm) * 1e-3 / 2.0, 1e-4)


def _furnace_temperature_C(t, p: Params):
    ramp = p.heating_rate_C_min / 60.0
    return min(p.T0_C + ramp*t, p.target_temp_C)


def _update_temperature_1d(T_nodes, T_env_K, dt, dx, p: Params):
    """Digital Twin用の簡易1D熱伝導。中心-表面温度差を半定量的に表す軽量モデル。"""
    n = len(T_nodes)
    if n <= 2:
        return np.full_like(T_nodes, T_env_K)

    k = max(float(p.thermal_conductivity_W_mK), 1e-3)
    rho_cp = max(float(p.density_bulk_kg_m3) * float(p.heat_capacity_J_kgK), 1.0)
    alpha = k / rho_cp

    # 炉径に対して試料が大きいほど実効熱伝達が落ちる簡略補正
    sample_size = max(float(p.sample_diameter_mm), float(p.sample_width_mm), float(p.sample_thickness_mm))
    furnace = max(float(p.furnace_inner_diameter_mm), sample_size + 1e-6)
    blockage = np.clip(sample_size / furnace, 0.0, 0.95)
    h_eff = max(float(p.heat_transfer_W_m2K), 1e-3) * (1.0 - 0.55 * blockage)

    # explicit scheme stability guard: substep internally if needed
    dt_stable = 0.35 * dx*dx / max(alpha, 1e-12)
    nsub = int(max(1, np.ceil(dt / max(dt_stable, 1e-6))))
    subdt = dt / nsub
    T = T_nodes.astype(float).copy()
    for _ in range(nsub):
        old = T.copy()
        T[1:-1] = old[1:-1] + alpha * subdt * (old[2:] - 2*old[1:-1] + old[:-2]) / (dx*dx)
        # 両表面で対流境界。中心対称ではなく「厚み方向の両面加熱」の近似。
        surf_coeff = h_eff / (rho_cp * max(dx, 1e-9))
        T[0] = old[0] + alpha * subdt * 2*(old[1] - old[0])/(dx*dx) + surf_coeff*subdt*(T_env_K - old[0])
        T[-1] = old[-1] + alpha * subdt * 2*(old[-2] - old[-1])/(dx*dx) + surf_coeff*subdt*(T_env_K - old[-1])
    return T


def _advance_sintering_state(T, rho, G, neck, dt, p: Params):
    Ds, Db, Dv = diffusivities(T, p)
    pore = max(1-rho, 0)
    fl = flags(T, rho, G, p)
    drho, active_model = densification_model(T, rho, G, pore, Ds, Db, Dv, fl, p)
    dG = grain_growth(T, rho, G, pore, p, fl)
    dX = neck_rate(G, Ds, Db, Dv, fl)
    rho_cap = float(np.clip(p.target_final_density, p.rho0 + 0.02, 0.999))
    rho = float(np.clip(rho + drho*dt, p.rho0, rho_cap))
    G = float(max(G + dG*dt, 0.005))
    neck = float(np.clip(neck + dX*dt, 0, 1))
    return rho, G, neck, Ds, Db, Dv, drho, dG, active_model, fl


def simulate(p: Params):
    # Research mode: 従来の0D平均場モデル。Digital Twin: 試料サイズを使う1D熱伝導＋焼結連成。
    if p.mode != "Digital Twin":
        dt = p.dt
        n = int(p.total_time_s//dt) + 1
        rho, G, neck = p.rho0, p.G0_um, 0.04
        rows = []
        for i in range(n):
            t = i*dt
            T = temp_profile(t, p)
            rho, G, neck, Ds, Db, Dv, drho, dG, active_model, fl = _advance_sintering_state(T, rho, G, neck, dt, p)
            rows.append({"t":t,"T_C":T-273.15,"T_surface_C":T-273.15,"T_center_C":T-273.15,"deltaT_C":0.0,
                         "rho":rho,"rho_surface":rho,"rho_center":rho,"rho_gradient":0.0,
                         "porosity":1-rho,"G_um":G,"neck":neck,
                         "Ds":Ds,"Db":Db,"Dv":Dv,"d_rho_dt":drho,"dG_dt":dG,"active_model":active_model,
                         **{k+"_flag":v for k,v in fl.items()}})
        return pd.DataFrame(rows)

    # Digital Twin: 1D through-thickness/radius thermal calculation with local sintering state per node.
    dt = max(p.dt, 0.5)
    nstep = int(p.total_time_s//dt) + 1
    nnode = int(np.clip(getattr(p, "thermal_nodes", 11), 5, 31))
    L = _geometry_half_length_m(p)
    # ここでは -L..+L を解くため全厚み/直径方向。dxは代表節点間隔。
    x = np.linspace(-L, L, nnode)
    dx = max(x[1] - x[0], 1e-6)
    T_nodes = np.full(nnode, p.T0_C + 273.15, dtype=float)
    rho_nodes = np.full(nnode, p.rho0, dtype=float)
    G_nodes = np.full(nnode, p.G0_um, dtype=float)
    neck_nodes = np.full(nnode, 0.04, dtype=float)
    rows = []

    for i in range(nstep):
        t = i*dt
        T_env_C = _furnace_temperature_C(t, p)
        # 炉内均熱性: 時間とともに弱い周期的ゆらぎとして反映。再現性のため決定論的。
        T_env_C += float(p.furnace_uniformity_C) * 0.15 * np.sin(2*np.pi*t/max(p.total_time_s, 1.0))
        T_env_K = T_env_C + 273.15
        T_nodes = _update_temperature_1d(T_nodes, T_env_K, dt, dx, p)

        # 電場/通電のジュール発熱は全体に加える。サイズが大きいほど抜熱が遅い簡略補正。
        if abs(p.E_V_m*p.J_A_m2) > 0:
            joule = np.clip(abs(p.E_V_m*p.J_A_m2)*1e-8, 0, 250)
            T_nodes += joule * dt / max(p.total_time_s, dt)

        Ds_list=[]; Db_list=[]; Dv_list=[]; drho_list=[]; dG_list=[]; model_list=[]; fl_list=[]
        for j in range(nnode):
            rho_nodes[j], G_nodes[j], neck_nodes[j], Ds, Db, Dv, drho, dG, active_model, fl = _advance_sintering_state(
                T_nodes[j], float(rho_nodes[j]), float(G_nodes[j]), float(neck_nodes[j]), dt, p
            )
            Ds_list.append(Ds); Db_list.append(Db); Dv_list.append(Dv); drho_list.append(drho); dG_list.append(dG)
            model_list.append(active_model); fl_list.append(fl)

        # 代表値: 平均、表面平均、中心値
        center_idx = nnode // 2
        surf_rho = float(0.5*(rho_nodes[0] + rho_nodes[-1]))
        center_rho = float(rho_nodes[center_idx])
        avg_rho = float(np.mean(rho_nodes))
        avg_G = float(np.mean(G_nodes))
        avg_neck = float(np.mean(neck_nodes))
        T_surf = float(0.5*(T_nodes[0] + T_nodes[-1]) - 273.15)
        T_center = float(T_nodes[center_idx] - 273.15)
        fl = fl_list[center_idx]
        rows.append({"t":t,"T_C":float(np.mean(T_nodes)-273.15),"T_surface_C":T_surf,"T_center_C":T_center,
                     "deltaT_C":abs(T_surf - T_center),
                     "rho":avg_rho,"rho_surface":surf_rho,"rho_center":center_rho,"rho_gradient":abs(surf_rho-center_rho),
                     "porosity":1-avg_rho,"G_um":avg_G,"neck":avg_neck,
                     "Ds":float(np.mean(Ds_list)),"Db":float(np.mean(Db_list)),"Dv":float(np.mean(Dv_list)),
                     "d_rho_dt":float(np.mean(drho_list)),"dG_dt":float(np.mean(dG_list)),
                     "active_model":model_list[center_idx],
                     **{k+"_flag":v for k,v in fl.items()}})
    return pd.DataFrame(rows)

def predict_properties(df, p: Params, second_phase_E_GPa=700, second_phase_H_GPa=20):
    out = df.copy()
    G = np.maximum(out['G_um'].values, 0.005)
    P = np.maximum(out['porosity'].values, 0)
    # Hall-Petch + inverse HP softening for nano region
    H_hp = p.H0_GPa + p.hp_k/np.sqrt(G)
    nano_soft = np.where(G < p.inverse_hp_threshold_um, G/p.inverse_hp_threshold_um, 1.0)
    H_dense = H_hp*nano_soft
    # Rice exponential porosity correction for hardness
    H_porous = H_dense*np.exp(-4.5*P)
    vf = p.second_phase_fraction
    H_voigt = (1-vf)*H_porous + vf*second_phase_H_GPa
    H_reuss = 1/((1-vf)/np.maximum(H_porous,1e-6) + vf/max(second_phase_H_GPa,1e-6))
    H_vrh = 0.5*(H_voigt+H_reuss)
    E_matrix = p.E0_GPa*np.exp(-3.2*P)
    E_voigt = (1-vf)*E_matrix + vf*second_phase_E_GPa
    E_reuss = 1/((1-vf)/np.maximum(E_matrix,1e-6) + vf/max(second_phase_E_GPa,1e-6))
    E_vrh = 0.5*(E_voigt + E_reuss)
    # rough toughness: porosity decreases, finer grains mildly increase up to a point
    KIC = p.KIC0_MPam05*np.exp(-2.2*P)*(1+0.08*np.log1p(1/np.maximum(G,0.01)))
    out['Hardness_GPa'] = H_vrh
    out['Elastic_Modulus_GPa'] = E_vrh
    out['KIC_MPa_m0.5'] = KIC
    return out
