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
    drive = max(1-rho, 0)**1.2
    rate = 2e-4*k*drive*geom*closed_penalty*pin_penalty*mechanism_factor
    rate *= max(0.05, 1-0.65*surface_loss)
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


def simulate(p: Params):
    # Research mode: standard dt. Digital Twin: smaller dt + more diagnostic outputs.
    dt = p.dt if p.mode == "Research" else max(p.dt/2, 0.5)
    n = int(p.total_time_s//dt) + 1
    rho, G, neck = p.rho0, p.G0_um, 0.04
    rows = []
    for i in range(n):
        t = i*dt
        T = temp_profile(t, p)
        # local Joule heating / digital twin only resolves stronger field influence
        if p.mode == "Digital Twin":
            T += np.clip(abs(p.E_V_m*p.J_A_m2)*1e-8, 0, 250)
        Ds, Db, Dv = diffusivities(T, p)
        pore = max(1-rho, 0)
        fl = flags(T, rho, G, p)
        drho, active_model = densification_model(T, rho, G, pore, Ds, Db, Dv, fl, p)
        dG = grain_growth(T, rho, G, pore, p, fl)
        dX = neck_rate(G, Ds, Db, Dv, fl)
        rho = float(np.clip(rho + drho*dt, p.rho0, 0.999))
        G = float(max(G + dG*dt, 0.005))
        neck = float(np.clip(neck + dX*dt, 0, 1))
        rows.append({"t":t,"T_C":T-273.15,"rho":rho,"porosity":1-rho,"G_um":G,"neck":neck,
                     "Ds":Ds,"Db":Db,"Dv":Dv,"d_rho_dt":drho,"dG_dt":dG,"active_model":active_model,
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
