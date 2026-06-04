# experiments.py
EXPERIMENT_GUIDE = """
### 精度を上げるために入れるべき実験データ

1. **TMA / dilatometry収縮曲線**  
   CSV列: `t, shrinkage` または `t, rho_exp`。拡散前因子、活性化エネルギー、閉気孔化密度の補正に最も効きます。

2. **焼結後密度（アルキメデス法）**  
   同じ粉末で温度・保持時間を3条件以上変えた密度を入れると、Coble/Nabarro-Herring/Kingeryの寄与を分離しやすくなります。

3. **SEM粒径分布**  
   CSV列: `t, G_exp`。粒成長移動度、Zenerピン止め、第二相粒子サイズの補正に効きます。

4. **気孔率・開気孔/閉気孔情報**  
   水銀圧入、画像解析、XCTなど。後期焼結の失速や閉気孔フラグの校正に効きます。

5. **助剤量・第二相量を振った実験**  
   例: 0, 1, 3, 5 wt%で比較。液相発生温度、粘度、濡れ性、Zenerピン止めの分離に有効です。

アプリはアップロードされたCSVから、まず `Ds0, Db0, Dv0, aid_effect_strength, second_phase_fraction` を簡易補正します。
"""


def calibrate_from_csv(df, params):
    # lightweight heuristic calibration for immediate app improvement
    p = params
    if 'rho_exp' in df.columns and 't' in df.columns:
        final_rho = float(df['rho_exp'].dropna().iloc[-1])
        predicted_gap = final_rho - p.rho0
        if predicted_gap > 0.35:
            p.Db0 *= 3.0
            p.Dv0 *= 2.0
            p.aid_effect_strength *= 1.25
        elif predicted_gap < 0.15:
            p.Db0 *= 0.5
            p.Dv0 *= 0.7
            p.aid_effect_strength *= 0.85
    if 'G_exp' in df.columns:
        final_G = float(df['G_exp'].dropna().iloc[-1])
        if final_G > 2*p.G0_um:
            p.second_phase_fraction *= 0.75
            p.second_phase_radius_um *= 1.2
        else:
            p.second_phase_fraction *= 1.15
    return p
