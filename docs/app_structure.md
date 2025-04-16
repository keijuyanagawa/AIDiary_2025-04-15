# AIDiary (app.py) コード構造図

```mermaid
graph TD
    A[Start / 開始] --> B(Import Libraries / ライブラリインポート);
    B --> C{Configure Environment / 環境設定};
    C --> D(Load .env / .envファイル読み込み);
    D --> E(Get API Keys & Supabase Credentials / APIキー・Supabase情報取得);
    E --> F{Configure AI Model / AIモデル設定};
    F -- API Key Found / キーあり --> G(Initialize Gemini Model / Geminiモデル初期化);
    F -- API Key Missing / キーなし --> H(AI Features Disabled / AI機能無効);
    G --> I{Configure Supabase Client / Supabaseクライアント設定};
    H --> I;
    I -- Credentials Found / 情報あり --> J(Initialize Supabase Client / Supabaseクライアント初期化);
    I -- Credentials Missing / 情報なし --> K(DB Features Disabled / DB機能無効);
    J --> L(Define DB Table Names / DBテーブル名定義);
    K --> L;
    L --> M(Initialize Streamlit Session State / Streamlitセッション状態初期化);
    M --> N(Set Streamlit Page Config / Streamlitページ設定);

    N --> O{Sidebar Navigation / サイドバーナビゲーション};
    O -- 日記を書く --> P(Set page='Diary' / ページを日記に設定);
    O -- 投稿を見る --> Q(Set page='View' / ページを投稿一覧に設定);
    O -- 感情を可視化 --> R(Set page='Visualize' / ページを可視化に設定);
    O -- カレンダーを見る --> S(Set page='Calendar' / ページをカレンダーに設定);
    P --> T{Rerun / アプリ再実行};
    Q --> T;
    R --> T;
    S --> T;

    T --> U{Main Content Area / メインコンテンツエリア};
    U -- Check Supabase Config / Supabase設定確認 --> V{If Supabase OK / OK?};
    U -- Check Supabase Config / Supabase設定確認 --> W[Show Supabase Error / エラー表示];
    W --> Z[End / 終了];
    V --> X{Route based on st.session_state.page / ページ状態で分岐};

    X -- page == 'Diary' --> DiaryPage;
    X -- page == 'View' --> ViewPage;
    X -- page == 'Visualize' --> VisualizePage;
    X -- page == 'Calendar' --> CalendarPage;

    subgraph DiaryPage ["📝 日記を書く (Diary)"]
        direction TB
        D1(Display Header / ヘッダー表示) --> D2(Get 'On This Day' Entries / 過去の同じ日取得);
        D2 --> D3(Display Date/Time Selection / 日時選択表示);
        D3 --> D4(Display Chat History / チャット履歴表示);
        D4 --> D5{Chat Input / チャット入力};
        D5 -- User Input / ユーザー入力あり --> D6(Append User Msg / ユーザー発言追加);
        D6 --> D7(Call get_ai_response / AI応答取得関数呼び出し);
        D7 -- Normal Response / 通常応答 --> D8(Display AI Response & Append / AI応答表示・追加);
        D7 -- Save Action / 保存指示 --> D9(Summarize & Save Diary / 会話要約・日記保存);
        D8 --> DiaryPageEnd(End Diary Interaction / 日記処理終了);
        D9 --> DiaryPageEnd;
        D5 -- No Input / ユーザー入力なし --> D10{Manual Save Button / 手動保存ボタン};
        D10 -- Clicked / クリック --> D11(Save User Messages as Diary / ユーザー発言を日記として保存);
        D10 -- Not Clicked / クリックなし --> DiaryPageEnd;
        D11 --> DiaryPageEnd;
    end

    subgraph ViewPage ["👀 投稿を見る (View)"]
        direction TB
        V1(Display Header / ヘッダー表示) --> V2(Display Filters Expander / フィルタ表示);
        V2 -- Apply Filters / フィルタ適用 --> V3(Update Session State & Rerun / 状態更新・再実行);
        V2 -- Clear Filters / フィルタ解除 --> V4(Reset Session State & Rerun / 状態リセット・再実行);
        V2 -- No Action / 操作なし --> V5(Call get_filtered_entries / フィルタされた投稿取得);
        V5 --> V6{Display Entries Loop / 投稿ループ表示};
        V6 -- For Each Entry / 各投稿 --> V7(Display Expander: Text, Analysis / 詳細表示: 本文, 分析);
        V7 --> V8{Delete Button / 削除ボタン};
        V8 -- Clicked / クリック --> V9(Call delete_entry & Rerun / 削除関数呼び出し・再実行);
        V6 -- End Loop / ループ終了 --> ViewPageEnd(End View Page / 投稿一覧処理終了);
    end

    subgraph VisualizePage ["📊 感情を可視化 (Visualize)"]
        direction TB
        Vis1(Display Header / ヘッダー表示) --> Vis2(Display Date Range Selection / 日付範囲選択表示);
        Vis2 --> Vis3(Call get_emotion_data / 感情データ取得);
        Vis3 --> Vis4(Convert to Pandas DataFrame / DataFrame変換);
        Vis4 --> Vis5(Map Emotion to Sentiment Value / 感情を数値に変換);
        Vis5 --> Vis6(Create Matplotlib Plot / Matplotlibグラフ作成);
        Vis6 --> Vis7(Display Plot using st.pyplot / グラフ表示);
        Vis7 --> Vis8(Optional: Display Data Table / データテーブル表示(任意));
        Vis8 --> VisualizePageEnd(End Visualize Page / 可視化処理終了);
    end

    subgraph CalendarPage ["📅 カレンダーを見る (Calendar)"]
        direction TB
        Cal1(Display Header / ヘッダー表示) --> Cal2(Call get_entry_dates / 投稿日付取得);
        Cal2 --> Cal3(Create Calendar Event List / カレンダーイベントリスト作成);
        Cal3 --> Cal4(Display Calendar using streamlit-calendar / カレンダー表示);
        Cal4 --> CalendarPageEnd(End Calendar Page / カレンダー処理終了);
    end

    subgraph FunctionDefs ["🛠️ Function Definitions / 関数定義"]
        direction TB
        DBFuncs(Database Functions / DB関数
add_diary_entry, update_entry_summary, add_emotion_record, get_*, delete_entry);
        AIFuncs(AI Functions / AI関数
generate_summary, analyze_emotion, summarize_chat_for_diary, get_ai_response);
    end

    DiaryPage --> Z;
    ViewPage --> Z;
    VisualizePage --> Z;
    CalendarPage --> Z;

    %% Link main pages to function calls (simplified)
    DiaryPage -- Calls --> DBFuncs;
    DiaryPage -- Calls --> AIFuncs;
    ViewPage -- Calls --> DBFuncs;
    VisualizePage -- Calls --> DBFuncs;
    CalendarPage -- Calls --> DBFuncs;

```

**図の説明:**

- この図は `app.py` の主要な処理の流れを示しています。
- 開始からライブラリのインポート、環境設定、Supabase/AIモデルの初期化、セッション状態の管理へと進みます。
- サイドバーのナビゲーションによって `st.session_state.page` が設定され、アプリが再実行（Rerun）されます。
- メインエリアでは、現在のページ状態に基づいて対応するページのロジックが実行されます。
- 各ページのサブグラフは、そのページ内の主要な処理ステップ（データの取得、表示、ユーザーインタラクション、関数呼び出しなど）を示しています。
- 主要なデータベース操作関数とAI関連関数は、別のサブグラフにまとめられています。
- 各要素には英語の説明と日本語の補足説明が付記されています。

**利用方法:**

- この Markdown ファイル (`app_structure.md`) を Mermaid をサポートするエディタ (VS Code の拡張機能など) やオンラインツール (Mermaid Live Editor など) で開くと、フローチャートとして視覚的に表示されます。 