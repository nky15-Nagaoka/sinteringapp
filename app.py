# sintering.app
import copy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import asdict

from materials_db import MATERIAL_PRESETS, REFERENCES_TEXT
from sintering_core import Params, simulate, predict_properties
from visualization import draw_microstructure
from experiments import EXPERIMENT_GUIDE, calibrate_from_csv

st.set_page_config(page_title="AI焼結シミュレーター", layout="wide")
st.title("AI焼結シミュレーター")
st.caption("Researchモード / Digital Twinモード対応：焼結機構、微細構造、実験フィードバック、簡易物性予測を統合")

@st.cache_data(show_spinner=False)
def run_cached(pdict):
    return simulate(Params(**pdict))

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
st.sidebar.caption("手で摘むと、右側の微細構造・時系列・物性が変化します。")
p.target_temp_C = st.sidebar.slider("焼結温度 [°C]", 500, 2200, int(p.target_temp_C), 10)
p.heating_rate_C_min = st.sidebar.slider("昇温速度 [°C/min]", 1.0, 50.0, 10.0, 0.5)
p.total_time_s = st.sidebar.slider("総時間 [s]", 300, 30000, int(max(p.hold_time_s, 3600)), 300)
p.rho0 = st.sidebar.slider("初期相対密度", 0.35, 0.75, float(p.rho0), 0.01)
p.G0_um = st.sidebar.slider("初期平均粒径 [µm]", 0.02, 10.0, float(p.G0_um), 0.01)
p.sintering_aid_fraction = st.sidebar.slider("焼結助剤/液相量", 0.0, 0.30, float(p.sintering_aid_fraction), 0.005)
p.second_phase_fraction = st.sidebar.slider("第二相・ナノ分散相量", 0.0, 0.30, float(p.second_phase_fraction), 0.005)
p.second_phase_radius_um = st.sidebar.slider("第二相半径 [µm]", 0.01, 2.0, float(p.second_phase_radius_um), 0.01)

with st.sidebar.expander("詳細パラメタ: 拡散・液相・雰囲気・電場"):
    p.dt = st.slider("dt [s]", 0.5, 20.0, 2.0, 0.5)
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
run = st.sidebar.button("シミュレーション実行", type="primary")
if 'last_df' not in st.session_state or run:
    with st.spinner("焼結過程を計算中..."):
        df = run_cached(asdict(p))
        dfp = predict_properties(df, p)
        st.session_state.last_df = df
        st.session_state.last_dfp = dfp
        st.session_state.last_params = asdict(p)

df = st.session_state.last_df
dfp = st.session_state.last_dfp
p_show = Params(**st.session_state.last_params)

# --- Main display tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["微細構造ビュー", "時系列", "物性予測", "実験フィードバック", "材料プリセット"])

with tab1:
    st.subheader("粒子 → ネック成長 → 緻密多結晶への模式図")
    c1, c2 = st.columns([2,1])
    with c2:
        time_pick = st.slider("観察時刻 [s]", float(df['t'].min()), float(df['t'].max()), float(df['t'].max()), step=max(float(df['t'].max()/200),1.0))
        row = df.iloc[(df['t']-time_pick).abs().argmin()]
        st.metric("相対密度", f"{row['rho']:.3f}")
        st.metric("気孔率", f"{row['porosity']:.3f}")
        st.metric("平均粒径", f"{row['G_um']:.2f} µm")
        st.write("有効モデル:", row['active_model'])
    with c1:
        fig = draw_microstructure(float(row['rho']), float(row['G_um']), float(row['porosity']), int(row['liquid_flag']), seed=int(row['t'])%999+1)
        st.pyplot(fig, clear_figure=True)
    st.caption("相対密度が上がると、円形粒子の集合から、粒界を持つ緻密多結晶模式図へ連続的に切り替わります。")

with tab2:
    st.subheader("密度・粒径・気孔率の時系列出力")
    plot_df = df.set_index('t')
    st.line_chart(plot_df[['rho','porosity','G_um']])
    with st.expander("拡散係数・フラグも表示"):
        st.line_chart(plot_df[['Ds','Db','Dv']])
        st.line_chart(plot_df[['surface_flag','gb_flag','lattice_flag','liquid_flag','closed_flag','abnormal_flag','thermal_runaway_flag']])
    st.download_button("時系列CSVをダウンロード", dfp.to_csv(index=False).encode('utf-8-sig'), file_name="sintering_result.csv")

with tab3:
    st.subheader("微細構造からの簡易物性予測")
    st.write("硬度: Hall-Petch / 逆Hall-Petch + Rice型気孔率補正 + Voigt-Reuss-Hill混合則")
    st.write("弾性率: 気孔率指数補正 + VRH混合則、靭性: 気孔率と粒径の簡易補正")
    st.line_chart(dfp.set_index('t')[['Hardness_GPa','Elastic_Modulus_GPa','KIC_MPa_m0.5']])
    last = dfp.iloc[-1]
    m1,m2,m3 = st.columns(3)
    m1.metric("最終硬度", f"{last['Hardness_GPa']:.2f} GPa")
    m2.metric("最終弾性率", f"{last['Elastic_Modulus_GPa']:.1f} GPa")
    m3.metric("最終靭性", f"{last['KIC_MPa_m0.5']:.2f} MPa m^0.5")

with tab4:
    st.subheader("実験からシミュレータを育てる")
    st.markdown(EXPERIMENT_GUIDE)
    st.info("対応CSV列例: `t,rho_exp,G_exp,porosity_exp,shrinkage`。アップロード後にサイドバーのパラメタが簡易補正されます。Digital Twinモードでは、今後ここをベイズ最適化/EnKFへ拡張する想定です。")
    if exp_df is not None:
        st.write("アップロード済みデータ")
        st.dataframe(exp_df.head(50))

with tab5:
    st.subheader("材料データ・カスタム材料")
    st.write("既定材料に加えて、学生ごとのテーマ材料はサイドバーの **カスタム材料** で入力できます。最小入力は、材料名、焼結様式、初期平均粒径、初期相対密度、代表焼結温度、保持時間、代表拡散係数/活性化エネルギーです。")
    st.json(preset)
    st.download_button(
        "この材料設定をCSVとして保存",
        pd.DataFrame([preset]).to_csv(index=False).encode('utf-8-sig'),
        file_name="custom_material_seed.csv",
        mime="text/csv",
    )
    st.markdown(REFERENCES_TEXT)
    st.warning("プリセット・カスタム値はいずれも研究開始用の初期値です。TMA密度曲線、SEM粒径、気孔率、助剤量依存性を入れて校正してください。")
