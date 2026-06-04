# sintering.app
import copy
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import asdict

from materials_db import MATERIAL_PRESETS, REFERENCES_TEXT
from sintering_core import Params, simulate, predict_properties
from experiments import EXPERIMENT_GUIDE, calibrate_from_csv

st.set_page_config(page_title="AI焼結シミュレーター", layout="wide")
st.title("AI焼結シミュレーター")
st.caption("Researchモード / Digital Twinモード対応：焼結機構、微細構造、実験フィードバック、簡易物性予測を統合")

@st.cache_data(show_spinner=False)
def run_cached(pdict):
    return simulate(Params(**pdict))

def _fmt_value(v):
    """Material summary display formatter."""
    if isinstance(v, float):
        if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.2e}"
        return f"{v:.3g}"
    return v

def render_material_summary(preset: dict):
    """Show material settings without raw JSON in the normal view."""
    category = preset.get("category", "-")
    model_hint = preset.get("model_hint", "-")
    initial_grain = preset.get("initial_grain_um", "-")
    rho0 = preset.get("rho0", "-")
    target_temp = preset.get("target_temp_C", "-")
    liquid_temp = preset.get("liquid_temp_C", "-")
    hold_s = preset.get("hold_time_s", "-")

    st.markdown("#### 現在の材料設定")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("焼結様式", str(category))
    c2.metric("推奨モデル", str(model_hint))
    c3.metric("初期粒径", f"{initial_grain} µm")
    c4.metric("初期相対密度", f"{rho0}")

    c5, c6, c7 = st.columns(3)
    c5.metric("焼結温度", f"{target_temp} °C")
    c6.metric("液相/反応開始", f"{liquid_temp} °C")
    try:
        c7.metric("保持時間", f"{float(hold_s)/60:.1f} min")
    except Exception:
        c7.metric("保持時間", str(hold_s))

    rows = [
        ("Ds0 表面拡散前因子", preset.get("Ds0", "-"), "m²/s"),
        ("Db0 粒界拡散前因子", preset.get("Db0", "-"), "m²/s"),
        ("Dv0 格子拡散前因子", preset.get("Dv0", "-"), "m²/s"),
        ("Qs 表面拡散活性化エネルギー", preset.get("Qs", "-"), "J/mol"),
        ("Qb 粒界拡散活性化エネルギー", preset.get("Qb", "-"), "J/mol"),
        ("Qv 格子拡散活性化エネルギー", preset.get("Qv", "-"), "J/mol"),
        ("第二相/ナノ分散相量", preset.get("second_phase_fraction", "-"), "fraction"),
        ("焼結助剤/液相量", preset.get("sintering_aid_fraction", "-"), "fraction"),
    ]
    df = pd.DataFrame(rows, columns=["項目", "値", "単位"])
    df["値"] = df["値"].map(_fmt_value)
    st.dataframe(df, use_container_width=True, hide_index=True)

    notes = preset.get("notes")
    if notes:
        st.info(notes)

    with st.expander("開発者向け: 材料設定の詳細JSON"):
        st.json(preset)


def _importance_color(stars: str) -> str:
    """Return badge background color for star importance."""
    text = str(stars)
    if "★★★★★" in text:
        return "#e03131"
    if "★★★★☆" in text:
        return "#f08c00"
    if "★★★☆☆" in text:
        return "#ffd43b"
    if "★★☆☆☆" in text:
        return "#74c0fc"
    return "#adb5bd"


def _render_colored_dataframe(df: pd.DataFrame, importance_col: str):
    """Render a colorful HTML table without pandas Styler.applymap.

    Streamlit Cloud can run newer pandas versions where Styler.applymap is removed.
    This function avoids pandas Styler completely and is therefore more portable.
    """
    import html

    header_cells = "".join(
        f"<th style='text-align:left;padding:8px 10px;background:#f1f3f5;border-bottom:1px solid #dee2e6;'>{html.escape(str(c))}</th>"
        for c in df.columns
    )
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col == importance_col:
                bg = _importance_color(str(val))
                fg = "#212529" if bg in ["#ffd43b", "#74c0fc"] else "white"
                cell = (
                    "<td style='padding:8px 10px;border-bottom:1px solid #edf2f7;'>"
                    f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
                    f"background:{bg};color:{fg};font-weight:700;white-space:nowrap;'>"
                    f"{html.escape(str(val))}</span></td>"
                )
            else:
                cell = f"<td style='padding:8px 10px;border-bottom:1px solid #edf2f7;vertical-align:top;'>{html.escape(str(val))}</td>"
            cells.append(cell)
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = (
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse;width:100%;font-size:0.92rem;'>"
        "<thead><tr>" + header_cells + "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_physics_guide():
    """Colorful glossary for students."""
    st.markdown("---")
    st.markdown("## このアプリでは以下の物理理論を使って計算しています")
    st.caption("各項目の意味、代表式、焼結挙動への効き方をまとめた学生向け用語集です。")

    guide_rows = [
        {"項目":"表面拡散係数 Ds","概要":"粒子表面に沿って原子・イオンが移動する速さ。ネックは成長するが、体積収縮への寄与は比較的小さい。","代表式":"Ds = Ds0 exp(-Qs/RT)","焼結に与える影響度":"★★★★☆","主な影響":"初期焼結、ネック成長、非緻密化成分"},
        {"項目":"粒界拡散係数 Db","概要":"結晶粒界に沿った高速拡散。中期焼結で密度上昇を支配しやすい。","代表式":"Db = Db0 exp(-Qb/RT)","焼結に与える影響度":"★★★★★","主な影響":"緻密化、収縮速度、Coble型焼結"},
        {"項目":"格子拡散係数 Dv","概要":"結晶粒内を通る体拡散。高温・後期焼結で重要になる。","代表式":"Dv = Dv0 exp(-Qv/RT)","焼結に与える影響度":"★★★★★","主な影響":"後期緻密化、Nabarro-Herring型焼結"},
        {"項目":"活性化エネルギー Qs, Qb, Qv","概要":"拡散を起こすために必要なエネルギー。大きいほど同じ温度で拡散しにくい。","代表式":"D = D0 exp(-Q/RT)","焼結に与える影響度":"★★★★★","主な影響":"温度依存性、最適焼結温度、昇温条件"},
        {"項目":"粒界エネルギー γb","概要":"粒界面積を減らそうとする駆動力。粒成長の主な原因になる。","代表式":"F ≈ 2γb/G","焼結に与える影響度":"★★★★☆","主な影響":"粒成長、異常粒成長、微細構造粗大化"},
        {"項目":"表面エネルギー γs","概要":"粉末粒子の表面積を減らそうとする焼結の基本駆動力。","代表式":"ΔG ∝ γs A","焼結に与える影響度":"★★★★★","主な影響":"初期焼結、ネック形成、表面積低下"},
        {"項目":"第二相ピン止め","概要":"第二相粒子やナノ分散相が粒界移動を妨げ、粒成長を抑える効果。","代表式":"Zener: Fpin ∝ f/r","焼結に与える影響度":"★★★★☆","主な影響":"粒成長抑制、微細粒維持、緻密化との競合"},
        {"項目":"焼結助剤","概要":"拡散促進、液相形成、粒界構造変化により焼結温度を下げる添加物。","代表式":"材料依存の補正係数","焼結に与える影響度":"★★★★☆","主な影響":"低温緻密化、液相焼結、粒成長促進/抑制"},
        {"項目":"液相生成","概要":"焼結中に液相が生じ、粒子再配列・溶解再析出により緻密化が進む。","代表式":"T ≥ T_liquid","焼結に与える影響度":"★★★★★","主な影響":"急速緻密化、Kingery型焼結、助剤系材料"},
        {"項目":"酸素分圧 pO2","概要":"酸化物中の空孔や荷電欠陥濃度を変え、拡散係数を変化させる。","代表式":"欠陥化学補正","焼結に与える影響度":"★★★☆☆","主な影響":"酸化物の緻密化、粒界状態、雰囲気依存性"},
        {"項目":"水蒸気分圧 pH2O","概要":"粒界構造や表面反応に影響し、特に非酸化物・助剤系で効く場合がある。","代表式":"雰囲気補正項","焼結に与える影響度":"★★☆☆☆","主な影響":"Si3N4/SiC系、粒界相、表面反応"},
        {"項目":"電場・ジュール発熱","概要":"FAST/SPS/Flash焼結で電流により局所加熱や欠陥濃度変化が起きる。","代表式":"q = J·E または q = J²ρe","焼結に与える影響度":"★★★☆☆","主な影響":"局所昇温、熱暴走、電場促進拡散"},
        {"項目":"開気孔→閉気孔転移","概要":"気孔が外部とつながった状態から孤立気孔になる転移。後期焼結を左右する。","代表式":"ρ ≈ 0.92 付近","焼結に与える影響度":"★★★★★","主な影響":"後期緻密化、残留気孔、最終密度"},
        {"項目":"異常粒成長","概要":"一部の粒だけが急成長する現象。閉気孔の取り込みや特性低下を起こす。","代表式":"G > Gcrit","焼結に与える影響度":"★★★★☆","主な影響":"粗大粒、硬度低下、ばらつき増加"},
        {"項目":"Cobleモデル","概要":"粒界拡散が支配的な焼結モデル。微粒・中温域でよく効く。","代表式":"dρ/dt ∝ Db/G³","焼結に与える影響度":"★★★★★","主な影響":"微粒粉末の緻密化、中期焼結"},
        {"項目":"Nabarro-Herringモデル","概要":"格子拡散が支配的な高温クリープ/焼結モデル。","代表式":"dρ/dt ∝ Dv/G²","焼結に与える影響度":"★★★★☆","主な影響":"高温焼結、後期焼結、粗粒材料"},
        {"項目":"Kingery液相焼結モデル","概要":"液相による再配列・溶解再析出・毛管力を考慮する液相焼結モデル。","代表式":"dρ/dt ∝ 液相量/粘度","焼結に与える影響度":"★★★★★","主な影響":"WC-Co、Si3N4助剤系、SiC助剤系"},
    ]
    guide_df = pd.DataFrame(guide_rows)

    def style_importance(val):
        text = str(val)
        if "★★★★★" in text:
            return "background-color:#e03131;color:white;font-weight:bold"
        if "★★★★☆" in text:
            return "background-color:#f08c00;color:white;font-weight:bold"
        if "★★★☆☆" in text:
            return "background-color:#ffd43b;color:#212529;font-weight:bold"
        if "★★☆☆☆" in text:
            return "background-color:#74c0fc;color:#102a43;font-weight:bold"
        return ""

    _render_colored_dataframe(guide_df, "焼結に与える影響度")

    with st.expander("影響度の読み方"):
        st.markdown(
            """
            - ★★★★★：密度・粒径・気孔率の予測に非常に強く効く
            - ★★★★☆：材料系によって強く効く
            - ★★★☆☆：条件によって重要
            - ★★☆☆☆：補助的だが、特定材料では無視できない
            """
        )

# --- Sidebar: unified controls ---
st.sidebar.header("1. モード・材料選択")
mode = st.sidebar.radio("計算モード", ["Research", "Digital Twin"], horizontal=True)
material_options = list(MATERIAL_PRESETS.keys()) + ["カスタム材料（学生・研究テーマ用）"]
mat_name = st.sidebar.selectbox("モデル材料", material_options)

# Base parameter object
p = Params()
p.mode = mode

# Preset or custom material editor.  The custom path asks for 6--7 literature-accessible
# values first, then maps them into the same simulator core so the web page remains unified.
if mat_name == "カスタム材料（学生・研究テーマ用）":
    st.sidebar.info("文献や実験ノートから拾いやすい最小パラメタで材料モデルを作成します。")
    with st.sidebar.expander("カスタム材料の基本情報", expanded=True):
        custom_name = st.text_input("材料名", "My Ceramic / Composite")
        custom_category = st.selectbox("焼結様式", ["固相焼結", "液相焼結", "固相＋液相", "その他・反応焼結"])
        custom_model_hint = st.selectbox(
            "推奨速度式",
            ["自動ハイブリッド", "Coble粒界拡散", "Nabarro-Herring格子拡散", "Kingery液相"],
            index=0,
        )
        custom_G0 = st.number_input("初期平均粒径 [µm]", min_value=0.005, max_value=100.0, value=0.50, format="%.4f")
        custom_rho0 = st.number_input("初期相対密度", min_value=0.30, max_value=0.80, value=0.55, format="%.3f")
        custom_T = st.number_input("代表焼結温度 [°C]", min_value=300.0, max_value=2400.0, value=1400.0, format="%.1f")
        custom_hold = st.number_input("代表保持時間 [min]", min_value=0.0, max_value=600.0, value=60.0, format="%.1f")
        custom_Db0 = st.number_input("代表拡散前因子 Db0 [m²/s]", min_value=1e-20, max_value=1e-3, value=1e-10, format="%.2e")
        custom_Qb = st.number_input("代表活性化エネルギー Qb [kJ/mol]", min_value=20.0, max_value=900.0, value=220.0, format="%.1f")

    with st.sidebar.expander("カスタム材料の任意補正"):
        custom_liquid_T = st.number_input("液相/反応開始温度 [°C]", min_value=300.0, max_value=9999.0,
                                          value=9999.0 if custom_category == "固相焼結" else max(500.0, custom_T-150.0), format="%.1f")
        custom_aid = st.number_input("焼結助剤/液相量 [体積または重量分率の近似]", min_value=0.0, max_value=0.50,
                                     value=0.00 if custom_category == "固相焼結" else 0.05, format="%.3f")
        custom_second = st.number_input("第二相/ナノ分散相量", min_value=0.0, max_value=0.50, value=0.02, format="%.3f")
        custom_second_r = st.number_input("第二相半径 [µm]", min_value=0.005, max_value=10.0, value=0.10, format="%.3f")
        custom_E = st.number_input("緻密体弾性率 E0 [GPa]", min_value=1.0, max_value=1000.0, value=300.0, format="%.1f")
        custom_H = st.number_input("緻密体硬度 H0 [GPa]", min_value=0.1, max_value=80.0, value=12.0, format="%.1f")

    preset = {
        "category": custom_category,
        "model_hint": custom_model_hint,
        "initial_grain_um": custom_G0,
        "rho0": custom_rho0,
        "target_temp_C": custom_T,
        "liquid_temp_C": custom_liquid_T,
        "hold_time_s": custom_hold * 60,
        # Use the literature-entered grain-boundary term as the anchor; derive rough surface/lattice seeds.
        "Ds0": custom_Db0 * 50,
        "Db0": custom_Db0,
        "Dv0": custom_Db0 * 0.01,
        "Qs": max(custom_Qb*1000*0.65, 1.0),
        "Qb": custom_Qb * 1000,
        "Qv": custom_Qb * 1000 * 1.55,
        "gamma_b": 1.0,
        "gamma_s": 1.5,
        "second_phase_fraction": custom_second,
        "second_phase_radius_um": custom_second_r,
        "sintering_aid_fraction": custom_aid,
        "aid_effect_strength": 1.0 if custom_category == "固相焼結" else 1.8,
        "E0_GPa": custom_E,
        "H0_GPa": custom_H,
        "KIC0_MPam05": 3.0,
        "notes": f"ユーザー定義材料: {custom_name}。入力された6--7個程度の文献値から初期推定。"
    }
    st.sidebar.success(f"カスタム材料: {custom_name}")
else:
    preset = MATERIAL_PRESETS[mat_name]
    st.sidebar.info(f"焼結様式: {preset['category']}\n\n推奨モデル: {preset['model_hint']}")

p.model = st.sidebar.selectbox("速度式モデル", ["自動ハイブリッド", "Coble粒界拡散", "Nabarro-Herring格子拡散", "Kingery液相"],
                               index=["自動ハイブリッド", "Coble粒界拡散", "Nabarro-Herring格子拡散", "Kingery液相"].index(preset['model_hint']) if preset['model_hint'] in ["自動ハイブリッド", "Coble粒界拡散", "Nabarro-Herring格子拡散", "Kingery液相"] else 0)

# load preset/custom defaults
for k, v in preset.items():
    if hasattr(p, k):
        setattr(p, k, v)
# material database uses an explicit name for the literature-seeded initial grain size
p.G0_um = float(preset.get('initial_grain_um', p.G0_um))
p.mode = mode

st.sidebar.header("2. イコライザー風 主要スライダー")
st.sidebar.caption("条件を設定した後、下の実行ボタンを押すと時系列・物性が更新されます。")
p.target_temp_C = st.sidebar.slider("焼結温度 [°C]", 500, 2200, int(p.target_temp_C), 10)
p.heating_rate_C_min = st.sidebar.slider("昇温速度 [°C/min]", 1.0, 50.0, 10.0, 0.5)
ramp_time_s_default = max((p.target_temp_C - p.T0_C) / max(p.heating_rate_C_min, 1e-6) * 60.0, 0.0)
default_total_s = int(max(ramp_time_s_default + float(getattr(p, "hold_time_s", 3600)), 3600))
st.sidebar.caption(f"昇温だけで約 {ramp_time_s_default/3600:.2f} h 必要です。総時間は最大3時間までに制限しています。")
time_unit = st.sidebar.radio("時間単位", ["分", "時間"], horizontal=True)

if time_unit == "分":
    total_min = st.sidebar.slider(
        "総時間 [min]",
        5,
        180,
        min(int(default_total_s / 60), 180),
        5
    )
    p.total_time_s = total_min * 60

else:
    total_h = st.sidebar.slider(
        "総時間 [h]",
        0.1,
        3.0,
        min(float(default_total_s / 3600), 3.0),
        0.1
    )
    p.total_time_s = total_h * 3600
if p.total_time_s < ramp_time_s_default:
    st.sidebar.warning("総時間が昇温時間より短いため、保持温度に到達する前に計算が終了します。密度はほとんど上がりません。")
p.rho0 = st.sidebar.slider("初期相対密度", 0.35, 0.75, float(p.rho0), 0.01)
p.G0_um = st.sidebar.slider("初期平均粒径 [µm]", 0.02, 10.0, float(p.G0_um), 0.01)
p.sintering_aid_fraction = st.sidebar.slider("焼結助剤/液相量", 0.0, 0.30, float(p.sintering_aid_fraction), 0.005)
p.second_phase_fraction = st.sidebar.slider("第二相・ナノ分散相量", 0.0, 0.30, float(p.second_phase_fraction), 0.005)
p.second_phase_radius_um = st.sidebar.slider("第二相半径 [µm]", 0.01, 2.0, float(p.second_phase_radius_um), 0.01)
st.sidebar.caption("密度が0.99付近まで上がらない場合は、焼結時間を長くするか、下の詳細設定で緻密化倍率を上げてください。")

with st.sidebar.expander("詳細パラメタ: 拡散・液相・雰囲気・電場"):
    p.dt = st.slider("dt [s]", 0.5, 20.0, 2.0, 0.5)

    p.target_final_density = st.slider("目標到達相対密度", 0.90, 0.999, float(p.target_final_density), 0.001)
    p.densification_scale = st.slider("緻密化計算倍率（TMA校正用）", 0.1, 200.0, float(p.densification_scale), 0.1)
    p.liquid_temp_C = st.slider("液相/反応開始温度 [°C]", 500, 2200, int(p.liquid_temp_C if p.liquid_temp_C < 9000 else 2200), 10)
    p.viscosity_log10_Pa_s_at_liquid = st.slider("液相粘度 log10(Pa s)", 0.0, 8.0, 3.0, 0.1)
    p.Ds0 = st.number_input("Ds0 [m²/s]", value=float(p.Ds0), format="%.2e")
    p.Db0 = st.number_input("Db0 [m²/s]", value=float(p.Db0), format="%.2e")
    p.Dv0 = st.number_input("Dv0 [m²/s]", value=float(p.Dv0), format="%.2e")
    p.Qs = st.number_input("Qs [J/mol]", value=float(p.Qs), format="%.1f")
    p.Qb = st.number_input("Qb [J/mol]", value=float(p.Qb), format="%.1f")
    p.Qv = st.number_input("Qv [J/mol]", value=float(p.Qv), format="%.1f")
    p.gamma_b = st.slider("粒界エネルギー γb [J/m²]", 0.1, 3.0, float(p.gamma_b), 0.05)
    p.gamma_s = st.slider("表面エネルギー γs [J/m²]", 0.1, 3.0, float(p.gamma_s), 0.05)
    p.aid_effect_strength = st.slider("焼結助剤効果係数", 0.0, 4.0, float(p.aid_effect_strength), 0.05)
    p.pO2_atm = st.number_input("酸素分圧 pO2 [atm]", value=0.21, format="%.2e")
    p.pH2O_atm = st.number_input("水蒸気分圧 pH2O [atm]", value=1e-4, format="%.2e")
    p.E_V_m = st.number_input("電場 E [V/m]", value=0.0, format="%.2e")
    p.J_A_m2 = st.number_input("電流密度 J [A/m²]", value=0.0, format="%.2e")
    p.open_closed_density = st.slider("開気孔→閉気孔転移密度", 0.85, 0.98, 0.92, 0.005)

with st.sidebar.expander("物性予測パラメタ"):
    p.E0_GPa = st.slider("母相弾性率 E0 [GPa]", 20.0, 800.0, float(p.E0_GPa), 5.0)
    p.H0_GPa = st.slider("母相硬度 H0 [GPa]", 1.0, 40.0, float(p.H0_GPa), 0.5)
    p.KIC0_MPam05 = st.slider("母相靭性 KIC0 [MPa m^0.5]", 0.3, 20.0, float(p.KIC0_MPam05), 0.1)

# --- Calibration ---
st.sidebar.header("3. 実験フィードバック")
upload = st.sidebar.file_uploader("実験CSVをアップロード", type=["csv"])
if upload is not None:
    exp_df = pd.read_csv(upload)
    p = calibrate_from_csv(exp_df, p)
    st.sidebar.success("CSVに基づき初期パラメタを簡易補正しました。")
else:
    exp_df = None

# --- Run simulation ---
run = st.sidebar.button("条件を反映してシミュレーション実行", type="primary")

# 重要: 初回表示やスライダー変更だけでは重いsimulate()を走らせない。
# これにより「何も動かしていないのに固まる」「スライダーを触るたび固まる」症状を避ける。
if run:
    with st.spinner("焼結過程を計算中..."):
        df = run_cached(asdict(p))
        dfp = predict_properties(df, p)
        st.session_state.last_df = df
        st.session_state.last_dfp = dfp
        st.session_state.last_params = asdict(p)

if 'last_df' not in st.session_state:
    st.info("左の条件を設定し、［条件を反映してシミュレーション実行］を押してください。初回表示時には計算を行わない軽量仕様です。")
    render_material_summary(preset)

    with st.expander("物理理論ガイドを表示する", expanded=False):
        render_physics_guide()

    with st.expander("シミュレーション精度向上のための実験データ項目を表示する", expanded=False):
        st.markdown(EXPERIMENT_GUIDE)

    st.stop()

df = st.session_state.last_df
dfp = st.session_state.last_dfp
p_show = Params(**st.session_state.last_params)

# --- Main display tabs ---
# 微細構造模式図は負荷が大きいため完全削除。物理計算は維持。
tab1, tab2, tab3, tab4, tab5 = st.tabs(["時系列", "物性予測", "実験フィードバック", "材料設定", "物理理論ガイド"])

with tab1:
    st.subheader("密度・粒径・気孔率の時系列出力")
    plot_df = df.set_index('t')
    st.line_chart(plot_df[['rho','porosity','G_um']])
    with st.expander("拡散係数・フラグも表示"):
        st.line_chart(plot_df[['Ds','Db','Dv']])
        st.line_chart(plot_df[['surface_flag','gb_flag','lattice_flag','liquid_flag','closed_flag','abnormal_flag','thermal_runaway_flag']])
    st.download_button("時系列CSVをダウンロード", dfp.to_csv(index=False).encode('utf-8-sig'), file_name="sintering_result.csv")

with tab2:
    st.subheader("微細構造からの簡易物性予測")
    st.write("硬度: Hall-Petch / 逆Hall-Petch + Rice型気孔率補正 + Voigt-Reuss-Hill混合則")
    st.write("弾性率: 気孔率指数補正 + VRH混合則、靭性: 気孔率と粒径の簡易補正")
    st.line_chart(dfp.set_index('t')[['Hardness_GPa','Elastic_Modulus_GPa','KIC_MPa_m0.5']])
    last = dfp.iloc[-1]
    m1,m2,m3 = st.columns(3)
    m1.metric("最終硬度", f"{last['Hardness_GPa']:.2f} GPa")
    m2.metric("最終弾性率", f"{last['Elastic_Modulus_GPa']:.1f} GPa")
    m3.metric("最終靭性", f"{last['KIC_MPa_m0.5']:.2f} MPa m^0.5")

with tab3:
    with st.expander("シミュレーション精度の向上のための実験データ項目を表示する", expanded=False):
        st.subheader("シミュレーション精度の向上のための実験データ項目")

        st.markdown("""
### このシミュレータの精度を更に向上させるために、あなたが実際に作る材料の以下の実験結果を入れることで、このシミュレータは精度が向上するように設計されています。

### 得られたデータは左の **「3. 実験フィードバック」** の項目から入力してください。
""")

        feedback_df = pd.DataFrame([
            ["TMA（収縮率曲線）", "★★★★★", "緻密化速度、拡散係数、活性化エネルギー"],
            ["アルキメデス密度", "★★★★★", "相対密度予測、最終密度、閉気孔化条件"],
            ["SEM粒径測定", "★★★★★", "粒成長モデル、第二相ピン止め、異常粒成長判定"],
            ["気孔率測定", "★★★★☆", "開気孔→閉気孔転移、残留気孔、後期焼結"],
            ["焼結助剤量依存性", "★★★★☆", "液相焼結係数、助剤効果係数、Kingeryモデル"],
            ["第二相量依存性", "★★★★☆", "Zenerピン止め、粒成長抑制係数"],
            ["XRD結晶相解析", "★★★☆☆", "相変態、反応焼結、液相/固相の判定"],
            ["SPS電流・電圧履歴", "★★★☆☆", "電場焼結、ジュール発熱、熱暴走判定"],
            ["酸素分圧依存実験", "★★☆☆☆", "欠陥化学補正、酸化物の拡散補正"],
            ["水蒸気雰囲気試験", "★★☆☆☆", "粒界構造補正、非酸化物・助剤系の雰囲気効果"],
        ], columns=["実験項目", "推奨度", "シミュレータで改善される項目"])

        _render_colored_dataframe(feedback_df, "推奨度")

        st.info("""
研究初心者向け推奨セット

① TMA収縮曲線  
② SEM粒径測定  
③ アルキメデス密度  

この3種類だけでも、緻密化速度・粒成長・最終密度の校正ができるため、十分に高精度化が期待できます。
""")

        with st.expander("CSV入力の例と補足"):
            st.markdown(EXPERIMENT_GUIDE)
            st.code("t,rho_exp,G_exp,porosity_exp,shrinkage\n0,0.55,0.50,0.45,0.00\n600,0.62,0.55,0.38,0.02", language="csv")

    if exp_df is not None:
        st.write("アップロード済みデータ")
        st.dataframe(exp_df.head(50))

with tab4:
    st.subheader("材料データ・カスタム材料")
    st.write("既定材料に加えて、学生ごとのテーマ材料はサイドバーの **カスタム材料** で入力できます。最小入力は、材料名、焼結様式、初期平均粒径、初期相対密度、代表焼結温度、保持時間、代表拡散係数/活性化エネルギーです。")
    render_material_summary(preset)
    st.download_button(
        "この材料設定をCSVとして保存",
        pd.DataFrame([preset]).to_csv(index=False).encode('utf-8-sig'),
        file_name="custom_material_seed.csv",
        mime="text/csv",
    )
    st.markdown(REFERENCES_TEXT)
    st.warning("プリセット・カスタム値はいずれも研究開始用の初期値です。TMA密度曲線、SEM粒径、気孔率、助剤量依存性を入れて校正してください。")

with tab5:
    with st.expander("物理理論ガイドを表示する", expanded=False):
        render_physics_guide()
