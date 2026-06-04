AI焼結シミュレーター 修正版

起動方法:
  pip install -r requirements.txt
  streamlit run app.py

修正内容:
  pandas/Streamlit Cloud のバージョン差で発生する Styler.applymap AttributeError を回避。
  物理理論ガイドはCSS付きHTMLテーブルで表示。
