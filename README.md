# AI焼結シミュレーター

Streamlitで動く焼結シミュレーションWebアプリです。

## 起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Gemini Digital Twin機能

Digital Twinモードでは、Google Gemini APIを任意で使い、以下を支援します。

- 実験結果の解釈
- 優先して補正すべきパラメタの提案
- 次に試す焼結条件の提案

APIキーは以下のいずれかで設定できます。

```bash
export GEMINI_API_KEY="your_api_key_here"
```

または、アプリのサイドバーで一時入力できます。APIキーはコードには保存しません。

## 注意

Gemini機能を使うと、シミュレーション要約・実験CSVの統計要約・主要パラメタがGoogle Gemini APIへ送信されます。機密データや未公開研究データの扱いには注意してください。
