# CHANGELOG

## 2025-04-16
- CSSによるスタイリング機能を追加
  - `static` ディレクトリと `style.css` ファイルを作成
  - `app.py` にCSSファイルを読み込む関数 (`load_css`) を追加
  - アプリ起動時に `static/style.css` を読み込むように設定
- UI要素のスタイルを調整 (`static/style.css`):
  - メインコンテンツエリアの上部パディングを調整 (`.st-emotion-cache-b499ls`)
  - 「日記を書く - AIとチャット」タイトルのフォントサイズを調整 (`h3#\\39 8ba5361`)
  - 日付・時刻選択欄を常に2カラム表示にし、カラム間の余白を均等に調整 (`div.stHorizontalBlock`, `div.stColumn`)
- ファイル構成の整理:
  - `docs/` ディレクトリを作成
  - 開発者向け参照ファイル (`app_structure.md`, `streamlit_default.css`, `streamlit_default.html`) を `docs/` に移動

## 2025-04-15
- Geminiモデルのバージョンを 'gemini-1.5-flash' から 'gemini-2.0-flash' に変更
  - AIの応答品質を向上させるため、最新モデルに更新 