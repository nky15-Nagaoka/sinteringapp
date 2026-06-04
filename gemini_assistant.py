# gemini_assistant.py
"""Gemini API helper for AI焼結シミュレーター.

This module is intentionally optional: the simulator runs without Gemini.
Set GEMINI_API_KEY / GOOGLE_API_KEY in the environment, Streamlit secrets,
or enter an API key in the Digital Twin tab/sidebar at runtime.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import pandas as pd

DEFAULT_MODEL = "gemini-2.5-flash"


def available_models() -> list[str]:
    return [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]


def get_api_key(explicit_key: Optional[str] = None, streamlit_secrets: Optional[dict] = None) -> Optional[str]:
    """Resolve Gemini API key without ever hardcoding it in the app files."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    if streamlit_secrets:
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                value = streamlit_secrets.get(name)
                if value:
                    return str(value)
            except Exception:
                pass
    return None


def make_summary_table(df: pd.DataFrame, dfp: pd.DataFrame) -> Dict[str, Any]:
    """Compact numerical summary to keep token use low and avoid sending full CSV."""
    if df is None or df.empty:
        return {}
    first = df.iloc[0]
    last = df.iloc[-1]
    max_rate_idx = int(df["d_rho_dt"].idxmax()) if "d_rho_dt" in df.columns and len(df) else 0
    peak = df.loc[max_rate_idx]
    final_props = dfp.iloc[-1].to_dict() if dfp is not None and not dfp.empty else {}
    cols = ["surface_flag", "gb_flag", "lattice_flag", "liquid_flag", "closed_flag", "abnormal_flag", "thermal_runaway_flag"]
    flag_fraction = {c: float(df[c].mean()) for c in cols if c in df.columns}
    return {
        "initial": {
            "t_s": float(first.get("t", 0)),
            "T_C": float(first.get("T_C", 0)),
            "rho": float(first.get("rho", 0)),
            "porosity": float(first.get("porosity", 0)),
            "G_um": float(first.get("G_um", 0)),
        },
        "final": {
            "t_s": float(last.get("t", 0)),
            "T_C": float(last.get("T_C", 0)),
            "rho": float(last.get("rho", 0)),
            "porosity": float(last.get("porosity", 0)),
            "G_um": float(last.get("G_um", 0)),
            "active_model": str(last.get("active_model", "")),
            "Hardness_GPa": float(final_props.get("Hardness_GPa", 0)) if final_props else None,
            "Elastic_Modulus_GPa": float(final_props.get("Elastic_Modulus_GPa", 0)) if final_props else None,
            "KIC_MPa_m0.5": float(final_props.get("KIC_MPa_m0.5", 0)) if final_props else None,
        },
        "peak_densification": {
            "t_s": float(peak.get("t", 0)),
            "T_C": float(peak.get("T_C", 0)),
            "d_rho_dt": float(peak.get("d_rho_dt", 0)),
            "active_model": str(peak.get("active_model", "")),
        },
        "flag_fraction": flag_fraction,
    }


def make_experimental_summary(exp_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if exp_df is None or exp_df.empty:
        return {"uploaded": False}
    summary: Dict[str, Any] = {"uploaded": True, "columns": list(exp_df.columns), "n_rows": int(len(exp_df))}
    for col in exp_df.columns:
        if pd.api.types.is_numeric_dtype(exp_df[col]):
            s = exp_df[col].dropna()
            if not s.empty:
                summary[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "last": float(s.iloc[-1]),
                }
    return summary


def build_digital_twin_prompt(
    material_name: str,
    preset: Dict[str, Any],
    params: Any,
    sim_summary: Dict[str, Any],
    exp_summary: Dict[str, Any],
    user_question: str,
) -> str:
    pdict = asdict(params) if hasattr(params, "__dataclass_fields__") else dict(params)
    safe_params = {k: pdict.get(k) for k in [
        "mode", "model", "target_temp_C", "heating_rate_C_min", "total_time_s", "dt",
        "rho0", "G0_um", "Ds0", "Db0", "Dv0", "Qs", "Qb", "Qv",
        "liquid_temp_C", "sintering_aid_fraction", "aid_effect_strength",
        "second_phase_fraction", "second_phase_radius_um", "target_final_density",
        "densification_scale", "pO2_atm", "pH2O_atm", "E_V_m", "J_A_m2"
    ] if k in pdict}
    return f"""
あなたはセラミックス焼結の研究支援AIです。以下の焼結シミュレーション結果と実験情報を見て、研究者・学生が次に何をすべきかを日本語で具体的に提案してください。

重要な制約:
- 断定しすぎず、シミュレータは半定量モデルであることを明示する。
- 物理的に不自然な点、パラメータ校正が必要な点を指摘する。
- TMA収縮曲線、アルキメデス密度、SEM粒径分布、画像気孔率、XRD/相分析、助剤量依存性など、具体的な追加実験を提案する。
- 最後に「次に試す3条件」を、焼結温度、保持時間、助剤量、初期粒径/成形密度の観点で提案する。
- 危険な助言や、根拠のない文献値の捏造はしない。

材料名: {material_name}
材料プリセット/メモ: {preset}
主要入力パラメタ: {safe_params}
シミュレーション要約: {sim_summary}
実験データ要約: {exp_summary}
ユーザーの追加質問: {user_question or '特になし'}

出力形式:
1. シミュレーション結果の解釈
2. 実験結果がある場合のズレの読み方 / ない場合に最低限測るべきデータ
3. 優先して補正すべきパラメタ
4. 次に試す3条件
5. 注意点
""".strip()


def ask_gemini(prompt: str, api_key: str, model_name: str = DEFAULT_MODEL) -> str:
    """Call Gemini via the official google-genai SDK."""
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-genai がインストールされていません。`pip install -r requirements.txt` を実行してください。") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.25,
            top_p=0.9,
            max_output_tokens=1800,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        return "Geminiから空の応答が返りました。モデル名、APIキー、利用制限を確認してください。"
    return text
