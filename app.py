import streamlit as st
import os
from datetime import datetime, date, timedelta, time
from dotenv import load_dotenv
import google.generativeai as genai
import json
import time as py_time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from streamlit_calendar import calendar
from supabase import create_client, Client

# --- Matplotlib 日本語フォント設定 (Windows) ---
try:
    # 利用可能な日本語フォントを試す (Yu Gothic を優先)
    plt.rcParams['font.family'] = 'Yu Gothic'
    print("Matplotlib font set to Yu Gothic.") # 確認用ログ
except Exception as e_font1:
    st.warning(f"日本語フォント 'Yu Gothic' の設定に失敗しました: {e_font1}. 次に 'MS Gothic' を試します。")
    try:
        plt.rcParams['font.family'] = 'MS Gothic'
        print("Matplotlib font set to MS Gothic.") # 確認用ログ
    except Exception as e_font2:
        st.warning(f"日本語フォント 'MS Gothic' の設定にも失敗しました: {e_font2}. システムに日本語フォントがインストールされているか確認してください。")
        # ここでさらに他のフォント ('Meiryo' など) を試すこともできます

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Configure AI Model ---
model = None
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini AI Model configured.")
    else:
        st.warning("GEMINI_API_KEY not found. AI features disabled.")
except Exception as e:
    st.error(f"Error configuring Gemini AI: {e}")
    model = None

# --- Configure Supabase ---
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client configured.")
    except Exception as e:
        st.error(f"Error configuring Supabase client: {e}")
else:
    st.warning("SUPABASE_URL or SUPABASE_KEY not found. Database features disabled.")

# --- Database Table Names ---
ENTRIES_TABLE = "entries"
EMOTIONS_TABLE = "emotions"

# --- Initialize Session State (Removed user/login state) ---
if 'page' not in st.session_state:
    st.session_state.page = 'Diary' # Default page
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False
if 'keyword_filter' not in st.session_state:
    st.session_state.keyword_filter = ""
if 'start_date_filter' not in st.session_state:
    st.session_state.start_date_filter = None
if 'end_date_filter' not in st.session_state:
    st.session_state.end_date_filter = None
if 'emotion_filter' not in st.session_state:
    st.session_state.emotion_filter = []
if 'viz_start_date' not in st.session_state:
    st.session_state.viz_start_date = date.today() - timedelta(days=30)
if 'viz_end_date' not in st.session_state:
    st.session_state.viz_end_date = date.today()
if 'messages' not in st.session_state:
    st.session_state.messages = [] # Chat messages

# --- Main App Logic ---
st.set_page_config(page_title="AIチャット日記＋感情可視化アプリ", 
                   layout="wide", page_icon="📝")

# Sidebar (Removed Login/Register/Logout)
with st.sidebar:
    st.title("AIチャット日記＋感情可視化アプリ")
    # Simplified Navigation
    st.sidebar.header("Navigation")
    if st.button("日記を書く", key="nav_write"):
         st.session_state.page = 'Diary'
         st.rerun()
    if st.button("投稿を見る", key="nav_view"):
         st.session_state.page = 'View'
         st.rerun()
    if st.button("感情を可視化する", key="nav_visualize"):
         st.session_state.page = 'Visualize'
         st.rerun()
    if st.button("カレンダーを見る", key="nav_calendar"):
         st.session_state.page = 'Calendar'
         st.rerun()

# --- Database Functions (Supabase) ---

def add_diary_entry(body, entry_datetime=None):
    """Adds a new diary entry for a specific datetime, returning the ID and creation time string."""
    if not supabase:
        st.error("Supabase client not configured.")
        return None, None
    try:
        dt_to_use = entry_datetime if entry_datetime else datetime.now()
        if dt_to_use > datetime.now():
             st.error("未来の日時では日記を保存できません。")
             return None, None

        timestamp_str = dt_to_use.isoformat() # Use ISO format for Supabase timestampz
        data, count = supabase.table(ENTRIES_TABLE)\
            .insert({"created_at": timestamp_str, "body": body, "summary": ""})\
            .execute()

        if data and len(data[1]) > 0:
            entry_id = data[1][0]['id']
            # Supabase returns the timestamp in a slightly different format sometimes
            # Re-format for consistency if needed, or use the returned one.
            # Using the original timestamp_str for now.
            return entry_id, timestamp_str
        else:
            # Log error details if possible
            st.error(f"データベースへの登録エラー: {data}")
            return None, None
    except Exception as e:
        st.error(f"データベースへの登録エラー: {e}")
        return None, None

def update_entry_summary(entry_id, summary):
    """Updates the summary for a given entry ID."""
    if not supabase:
        st.error("Supabase client が設定されていません。")
        return False
    try:
        data, count = supabase.table(ENTRIES_TABLE)\
            .update({"summary": summary})\
            .eq("id", entry_id)\
            .execute()
        # Check if update was successful (count might be useful, or check data content)
        return True # Assuming success if no exception
    except Exception as e:
        st.error(f"データベースへの更新エラー: {e}")
        return False

def add_emotion_record(entry_id, recorded_at, emotion_label, emotion_score=None):
    """Adds an emotion record linked to a diary entry."""
    if not supabase:
        st.error("Supabase client が設定されていません。")
        return False
    try:
        # Ensure recorded_at is in ISO format if it's not already
        recorded_at_iso = recorded_at
        if isinstance(recorded_at, str):
            # Attempt to parse if it's a string, otherwise assume it's datetime
             try:
                 dt_obj = datetime.strptime(recorded_at, "%Y-%m-%d %H:%M:%S")
                 recorded_at_iso = dt_obj.isoformat()
             except ValueError:
                 # If already ISO or different format, try using directly
                 pass
        elif isinstance(recorded_at, datetime):
            recorded_at_iso = recorded_at.isoformat()

        data, count = supabase.table(EMOTIONS_TABLE)\
            .insert({"entry_id": entry_id, "recorded_at": recorded_at_iso, "emotion_label": emotion_label, "emotion_score": emotion_score})\
            .execute()
        return True # Assuming success if no exception
    except Exception as e:
        st.error(f"データベースへの感情記録の登録エラー: {e}")
        return False

# --- Function to get distinct emotions for filtering ---
def get_distinct_emotions():
    """Fetches distinct emotion labels recorded."""
    if not supabase:
        st.error("Supabase client が設定されていません。")
        return []
    emotions = []
    try:
        # Use Supabase RPC or complex query if direct distinct isn't straightforward
        # For simplicity, fetching all and getting unique values in Python
        data, count = supabase.table(EMOTIONS_TABLE)\
            .select("emotion_label")\
            .neq("emotion_label", "") \
            .not_.is_("emotion_label", "null") \
            .execute()

        if data and len(data[1]) > 0:
            all_labels = [row['emotion_label'] for row in data[1]]
            emotions = sorted(list(set(all_labels)))
        else:
            # Handle potential errors in data structure
            if isinstance(data, tuple) and len(data)>0 and isinstance(data[0],dict) and data[0].get('message'):
               st.error(f"感情を取得する際のデータベースエラー: {data[0]['message']}")
            elif data and len(data) > 1 and not data[1]: # Success but empty result
                pass # No emotions found is not an error state
            else:
               st.error(f"不明なエラー: {data}")

    except Exception as e:
        st.error(f"感情ラベルの取得エラー: {e}")
    return emotions


# --- Modified DB Function to Fetch Filtered Entries ---
def get_filtered_entries(keyword=None, start_date=None, end_date=None, selected_emotions=None):
    """Fetches diary entries, applying optional filters."""
    if not supabase:
        st.error("Supabase client not configured.")
        return []
    entries = []
    try:
        query = supabase.table(ENTRIES_TABLE).select("""
            id, created_at, body, summary,
            emotions ( emotion_label, emotion_score )
        """)

        # Add filters dynamically
        if keyword:
            # Use 'ilike' for case-insensitive search
            query = query.ilike("body", f"%{keyword}%")

        if start_date:
            start_datetime_iso = datetime.combine(start_date, time.min).isoformat()
            query = query.gte("created_at", start_datetime_iso)

        if end_date:
            end_datetime_iso = datetime.combine(end_date, time.max).isoformat()
            query = query.lte("created_at", end_datetime_iso)

        # Filtering by emotion requires a join or subquery logic, which is complex here.
        # Fetch all matching other criteria first, then filter in Python if needed,
        # or use a Supabase RPC function for advanced filtering.
        # Simple approach: Fetch all and filter post-query if selected_emotions.
        query = query.order("created_at", desc=True)
        data, count = query.execute()

        if data and len(data[1]) > 0:
            raw_entries = data[1]
            # Process entries, potentially flattening emotion data
            for entry in raw_entries:
                # Extract the *first* associated emotion for simplicity in display
                # Modify this if multiple emotions per entry need handling differently
                emotion_info = entry.get('emotions')
                entry_emotion_label = None
                entry_emotion_score = None
                if isinstance(emotion_info, list) and len(emotion_info) > 0:
                    # Check if created_at matches recorded_at (if that link is intended)
                    # Simplified: just take the first emotion record found for the entry
                     first_emotion = emotion_info[0]
                     entry_emotion_label = first_emotion.get('emotion_label')
                     entry_emotion_score = first_emotion.get('emotion_score')


                # Filter by selected emotions (post-query filtering)
                if selected_emotions:
                    if entry_emotion_label in selected_emotions:
                         entries.append({
                             'id': entry['id'],
                             'created_at': entry['created_at'],
                             'body': entry['body'],
                             'summary': entry['summary'],
                             'emotion_label': entry_emotion_label,
                             'emotion_score': entry_emotion_score
                         })
                else: # No emotion filter applied
                      entries.append({
                         'id': entry['id'],
                         'created_at': entry['created_at'],
                         'body': entry['body'],
                         'summary': entry['summary'],
                         'emotion_label': entry_emotion_label,
                         'emotion_score': entry_emotion_score
                     })

        else:
            # Handle potential errors
            if isinstance(data, tuple) and len(data)>0 and isinstance(data[0],dict) and data[0].get('message'):
               st.error(f"データベースへのフィルター適用エラー: {data[0]['message']}")
            elif data and len(data) > 1 and not data[1]:
                pass # No entries found is valid
            else:
               st.error(f"不明なエラー: {data}")


    except Exception as e:
        st.error(f"データベースへのフィルター適用エラー: {e}")
    return entries


# --- New DB Function for Visualization ---
def get_emotion_data(start_date=None, end_date=None):
    """Fetches emotion data within a date range."""
    if not supabase:
        st.error("Supabase client が設定されていません。")
        return []
    emotion_data = []
    try:
        query = supabase.table(EMOTIONS_TABLE)\
                    .select("recorded_at, emotion_label, emotion_score")

        if start_date:
            start_datetime_iso = datetime.combine(start_date, time.min).isoformat()
            query = query.gte("recorded_at", start_datetime_iso)
        if end_date:
            end_datetime_iso = datetime.combine(end_date, time.max).isoformat()
            query = query.lte("recorded_at", end_datetime_iso)

        # Ensure data is ordered by time for plotting
        query = query.order("recorded_at", desc=False) # Ascending for plots

        data, count = query.execute()

        if data and len(data[1]) > 0:
            emotion_data = data[1] # Already in the desired format
        else:
             # Handle potential errors
            if isinstance(data, tuple) and len(data)>0 and isinstance(data[0],dict) and data[0].get('message'):
               st.error(f"データベースへの感情データの取得エラー: {data[0]['message']}")
            elif data and len(data) > 1 and not data[1]:
                pass # No data found is valid
            else:
               st.error(f"不明なエラー: {data}")

    except Exception as e:
        st.error(f"データベースへの感情データの取得エラー: {e}")
    return emotion_data


# --- New DB Function for Calendar ---
def get_entry_dates():
    """Fetches distinct dates (YYYY-MM-DD) where entries exist."""
    if not supabase:
        st.error("Supabase client が設定されていません。")
        return []
    entry_dates = []
    try:
        # This might be inefficient; consider an RPC function in Supabase
        # to get distinct dates directly from the timestamp.
        data, count = supabase.table(ENTRIES_TABLE)\
            .select("created_at")\
            .order("created_at", desc=False)\
            .execute()

        if data and len(data[1]) > 0:
            all_timestamps = [row['created_at'] for row in data[1]]
            # Extract date part and get unique values
            all_dates = [ts.split('T')[0] for ts in all_timestamps if ts] # Basic split assuming ISO format
            entry_dates = sorted(list(set(all_dates)))
        else:
            # Handle potential errors
            if isinstance(data, tuple) and len(data)>0 and isinstance(data[0],dict) and data[0].get('message'):
               st.error(f"データベースへの投稿日付の取得エラー: {data[0]['message']}")
            elif data and len(data) > 1 and not data[1]:
                pass # No dates found is valid
            else:
               st.error(f"不明なエラー: {data}")

    except Exception as e:
        st.error(f"データベースへの投稿日付の取得エラー: {e}")
    return entry_dates


# --- New DB Function for "On This Day" ---
def get_on_this_day_entries():
    """Fetches entries from the same month/day in previous years."""
    if not supabase:
        st.error("Supabase client が設定されていません。")
        return []
    entries = []
    try:
        today = date.today()
        current_month_day = today.strftime("%m-%d") # Format MM-DD
        current_year = today.year

        # Filtering by month/day and excluding current year directly in Supabase
        # might require specific PostgreSQL date functions via RPC or complex filters.
        # Simpler approach: Fetch recent years' data and filter in Python.
        # Fetch data from previous years (e.g., last 5 years for performance)
        start_date_past = date(current_year - 5, 1, 1) # Example: look back 5 years
        start_iso = start_date_past.isoformat()

        query = supabase.table(ENTRIES_TABLE).select("""
            id, created_at, body, summary,
            emotions ( emotion_label, emotion_score )
        """).lt("created_at", f"{current_year}-01-01T00:00:00") # Exclude current year

        query = query.order("created_at", desc=True)
        data, count = query.execute()

        if data and len(data[1]) > 0:
            all_past_entries = data[1]
            for entry in all_past_entries:
                try:
                    # Parse ISO timestamp and check month/day
                    created_dt = datetime.fromisoformat(entry['created_at'].replace('Z', '+00:00'))
                    if created_dt.strftime("%m-%d") == current_month_day:
                        # Extract emotion data similarly to get_filtered_entries
                        emotion_info = entry.get('emotions')
                        entry_emotion_label = None
                        entry_emotion_score = None
                        if isinstance(emotion_info, list) and len(emotion_info) > 0:
                            first_emotion = emotion_info[0]
                            entry_emotion_label = first_emotion.get('emotion_label')
                            entry_emotion_score = first_emotion.get('emotion_score')

                        entries.append({
                             'id': entry['id'],
                             'created_at': entry['created_at'], # Keep original string
                             'body': entry['body'],
                             'summary': entry['summary'],
                             'emotion_label': entry_emotion_label,
                             'emotion_score': entry_emotion_score
                         })
                except (ValueError, TypeError) as parse_error:
                    print(f"Skipping entry due to date parse error: {entry['created_at']}, Error: {parse_error}")
                    continue # Skip entries with unexpected date format
        else:
             # Handle potential errors
            if isinstance(data, tuple) and len(data)>0 and isinstance(data[0],dict) and data[0].get('message'):
               st.error(f"データベースへの'On This Day'投稿の取得エラー: {data[0]['message']}")
            elif data and len(data) > 1 and not data[1]:
                pass # No past entries is valid
            else:
               st.error(f"不明なエラー: {data}")

    except Exception as e:
        st.error(f"データベースへの'On This Day'投稿の取得エラー: {e}")
    return entries # Return entries matching the criteria


# --- New DB Function to Delete Entry ---
def delete_entry(entry_id):
    """Deletes a diary entry and associated emotion records by entry ID."""
    if not supabase:
        st.error("Supabase client が設定されていません。")
        return False
    try:
        # Need to delete from emotions table first due to potential foreign key,
        # or configure CASCADE DELETE in Supabase. Assuming manual deletion order.
        _, count_emotions = supabase.table(EMOTIONS_TABLE).delete().eq("entry_id", entry_id).execute()
        # Check count_emotions or errors if needed

        # Delete the main entry
        _, count_entries = supabase.table(ENTRIES_TABLE).delete().eq("id", entry_id).execute()
        # Check count_entries if deletion was successful

        print(f"Attempted deletion for entry {entry_id}.") # For debugging
        return True # Assume success if no exception
    except Exception as e:
        st.error(f"データベースへの投稿の削除エラー: {entry_id}: {e}")
        return False 

# --- AI Helper Functions ---
def generate_summary(text):
    # Keep this function to generate a very short summary for list view/expanders
    """Generates a very short (1-sentence) summary using the Gemini model."""
    if not model:
        return "(AIモデルが設定されていません)"
    try:
        prompt = f"以下のテキストを1文で簡潔に要約してください。:\n\n{text}"
        response = model.generate_content(prompt)
        # Add basic safety check for response structure
        if response and hasattr(response, 'text'):
           return response.text.strip()
        else:
           st.error(f"AI要約の生成エラー: {response}")
           return "(要約の生成に失敗しました: 無効なAI応答)"
    except Exception as e:
        st.error(f"短い要約の生成エラー: {e}")
        return "(要約の生成に失敗しました)"

def analyze_emotion(text):
    """Analyzes emotion using the Gemini model, returning a dict {label: str, score: float or None}.
       Score represents sentiment polarity from -1.0 (negative) to 1.0 (positive).
    """
    if not model:
        return {"label": "Unknown", "score": None}
    try:
        # Modified prompt to ask for sentiment polarity score (-1.0 to 1.0)
        prompt = f"""
        以下のテキストの内容を分析し、最も表現されている主要な感情とそのポジティブ/ネガティブ度合いを判断してください。
        1. 感情ラベルは以下のいずれかから選択してください: Positive, Negative, Neutral, Joy, Sadness, Anger, Surprise, Fear.
        2. 感情のポジティブ/ネガティブ度合いを示すセンチメントスコアを、-1.0 (非常にネガティブ) から +1.0 (非常にポジティブ) の範囲で評価してください (ニュートラルは0.0に近い値)。
        結果は必ず以下のJSON形式で返してください:
        {{"emotion_label": "<選択した感情ラベル>", "emotion_score": <センチメントスコア>}}

        テキスト:
        {text}
        """
        response = model.generate_content(prompt)
        # Add safety checks for response structure before parsing
        if not (response and hasattr(response, 'text')):
            st.error(f"AI感情分析の生成エラー: {response}")
            return {"label": "Error", "score": None}

        try:
            # Ensure the response text is clean JSON
            response_text = response.text.strip()
            # Basic cleaning: remove potential markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            result = json.loads(response_text)
            if isinstance(result, dict) and 'emotion_label' in result and 'emotion_score' in result:
                score = result.get('emotion_score')
                # Validate score is float or int, and within rough expected range (-1 to 1)
                if score is not None:
                   try:
                        score = float(score)
                        # Optional: Clamp score to -1.0 to 1.0 just in case
                        score = max(-1.0, min(1.0, score))
                   except (ValueError, TypeError):
                        score = None # Set score to None if invalid type/conversion fails
                else:
                    score = None # Handle explicitly null score

                return {"label": str(result['emotion_label']), "score": score}
            else:
                st.warning(f"AIフォーマットエラーまたはJSONのキーが不足しています: {response_text}")
                label = result.get('emotion_label', "Unknown") if isinstance(result, dict) else "Unknown"
                return {"label": str(label), "score": None} # Fallback score to None
        except json.JSONDecodeError:
            st.warning(f"AI応答のJSONパースエラー: {response.text}")
            # Simple fallback based on text content (less reliable)
            label = "Unknown"
            text_lower = response.text.lower()
            possible_labels = ["positive", "negative", "neutral", "joy", "sadness", "anger", "surprise", "fear"]
            for pl in possible_labels:
                if pl in text_lower:
                    label = pl.capitalize()
                    break
            return {"label": label, "score": None}
        except Exception as e:
             st.error(f"AI感情応答の内容の処理エラー: {e}")
             return {"label": "Error", "score": None}
    except Exception as e:
        st.error(f"AI感情分析の呼び出しエラーまたは応答の受け取りエラー: {e}")
        return {"label": "Error", "score": None}


# --- Function to Summarize Chat for Diary ---
def summarize_chat_for_diary(chat_history):
    """Summarizes the chat conversation into a diary entry from the user's perspective."""
    if not model:
        return "(AIによる日記生成に失敗しました)"

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    # Prompt doesn't strongly depend on a specific user, can remain similar
    prompt = f"""
    以下のユーザーとAIアシスタントの会話履歴を基に、ユーザー視点での自然な日記を生成してください。
    会話の内容を整理・要約し、ユーザー（"user"の発言者）が一人称（私）で書いたようなスタイルで記述してください。
    特にユーザーが話した出来事、考え、感情を中心にまとめてください。

    会話履歴:
    {history_text}

    生成する日記:
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
           return response.text.strip()
        else:
           st.error(f"AI日記要約の生成エラー: {response}")
           user_messages = [m["content"] for m in chat_history if m["role"] == "user"]
           return "\n".join(user_messages) if user_messages else "(日記の生成に失敗しました: 無効なAI応答)"
    except Exception as e:
        st.error(f"日記の生成中にエラーが発生しました: {e}")
        user_messages = [m["content"] for m in chat_history if m["role"] == "user"]
        return "\n".join(user_messages) if user_messages else "(日記の生成に失敗しました)"

def get_ai_response(chat_history):
    """Generates an AI response or triggers diary save based on the chat history."""
    if not model:
        return "申し訳ありません、現在AIアシスタントが利用できません。"

    # --- Check for Save Confirmation ---
    last_user_message = chat_history[-1]["content"] if chat_history and chat_history[-1]["role"] == "user" else ""
    last_ai_message = chat_history[-2]["content"] if len(chat_history) > 1 and chat_history[-2]["role"] == "assistant" else ""

    save_suggestion_keywords = ["保存しますか", "日記にしますか", "記録しますか"]
    agreement_keywords = ["はい", "うん", "お願い", "保存して", "いいよ", "わかった", "そうして", "頼む"]

    ai_suggested_save = any(keyword in last_ai_message for keyword in save_suggestion_keywords)
    user_agreed = any(keyword in last_user_message.lower() for keyword in agreement_keywords)

    if ai_suggested_save and user_agreed:
        print("ユーザーが保存を同意しました。会話を要約して日記を作成しています...")
        with st.spinner("会話を要約して日記を作成しています..."):
            diary_body = summarize_chat_for_diary(chat_history[:-1]) # 同意メッセージを除外
        # 要約に失敗したかどうかを確認
        if diary_body.startswith("(AIによる日記生成に失敗しました)") or diary_body.startswith("(日記の生成に失敗しました"):
             st.error("AIによる日記の要約に失敗したため、保存できませんでした。")
             return "申し訳ありません、日記の要約に失敗しました。" # Return an error message instead of the action dict
        else:
             return {"action": "save", "body": diary_body}
    # --- End Check for Save Confirmation ---

    # --- Generate Normal Chat Response ---
    # System prompt remains the same as it defines the AI's persona
    system_prompt = (
        "あなたはユーザーの日記作成をサポートする、親切で共感的なAIアシスタントです。あなたの役割は、ユーザーがリラックスして、その日の出来事、考え、感情について自由に、そして深く話せるように導くことです。"
        "会話を単なる質問応答ではなく、自然で温かい対話にしてください。\n\n"
        "## 対話の進め方:\n"
        "1. **傾聴と共感:** まずユーザーの話を注意深く聞き、内容を受け止め、共感を示してください。（例: 「そうだったんですね」「それは大変でしたね」「お気持ちお察しします」「それは素敵な体験でしたね！」）\n"
        "2. **要約と確認 (時々):** ユーザーの話が少し長くなったら、「〇〇があったんですね。それで、△△と感じた、ということでしょうか？」のように短く要約・確認すると、ユーザーは理解されていると感じ安心します。\n"
        "3. **多様な質問で深掘り:** 共感や確認の後、画一的にならないよう、様々な角度から質問を投げかけ、ユーザーの内面を引き出してください。\n"
        "    - **感情の理由・背景:** 「どうしてそう感じたのですか？」「何かきっかけがあったのでしょうか？」\n"
        "    - **具体的な状況:** 「その時、周りはどんな様子でしたか？」「誰か他にいましたか？」\n"
        "    - **思考・学習:** 「その経験から何か学びましたか？」「今振り返ってみてどう思いますか？」\n"
        "    - **未来・希望:** 「これからどうしたいですか？」「次に期待することはありますか？」\n"
        "    - **別の視点 (慎重に):** 「もし違う状況だったら、どうなっていたと思いますか？」\n"
        "    - **単純な促し:** 「それで、どうなりましたか？」「他には何かありましたか？」\n"
        "4. **自然な流れ:** 質問攻めにせず、会話の流れに合わせて自然なタイミングで質問してください。時には相槌や短い感想だけでも構いません。\n"
        "5. **記憶の活用:** 可能であれば、会話の前の内容に触れて、「先ほど〇〇とおっしゃっていましたが、それと関連はありますか？」のように繋げてみてください。\n"
        "6. **簡潔な応答:** あなた自身の応答は簡潔に、1〜3文程度にしてください。主役はユーザーです。\n\n"
        "## 日記保存の提案:\n"
        "会話が十分深まったと感じたら（例: 会話が5往復以上、またはユーザーの発言文字数合計が200字を超えた場合）、自然な流れで日記の保存を提案してください。\n"
        "例: 「たくさんお話しいただきありがとうございます。ここまでの内容を日記として記録しておきましょうか？」『ここまでの内容を日記として保存しますか？』のように明確に疑問符をつけて提案してください。"
    )
    api_history = []
    user_char_count = 0
    turn_count = 0
    for msg in chat_history:
        role = "model" if msg["role"] == "assistant" else "user" # Gemini API uses 'user' and 'model'
        api_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        if msg["role"] == "user": # Count user messages based on original role
            user_char_count += len(msg["content"])
            turn_count += 1

    should_suggest_save = turn_count >= 5 or user_char_count > 200

    try:
        # Construct conversation history for Gemini API
        # The system prompt can be added as the first user message or via specific API parameters if available
        conversation_to_send = [{"role": "user", "parts": [{"text": system_prompt}]}]
        # Add a starting assistant message to guide the conversation structure
        conversation_to_send.append({"role": "model", "parts": [{"text": "はい、こんにちは！今日はどんな一日でしたか？お手伝いできることがあれば教えてください。"}]})
        # Append the actual chat history
        conversation_to_send.extend(api_history)


        response = model.generate_content(conversation_to_send)
        response_text = response.text.strip() if response and hasattr(response, 'text') else "申し訳ありません、応答を生成できませんでした。"

        # Check if AI should have suggested saving but didn't (simple check)
        # This logic might need refinement based on AI behavior
        if should_suggest_save and not any(keyword in response_text for keyword in save_suggestion_keywords):
            # Optionally add the suggestion if the AI missed it
            # response_text += "\n\nここまでの内容を日記として保存しますか？"
            pass # For now, just proceed with the AI's response

        return response_text

    except Exception as e:
        st.error(f"AI応答の生成中にエラーが発生しました: {e}")
        # Attempt to access response details for debugging, e.g., safety feedback
        try:
             if response and response.prompt_feedback:
                  st.error(f"プロンプトフィードバック: {response.prompt_feedback}")
        except (AttributeError, NameError):
             pass # フィードバックまたは応答オブジェクトが利用できない場合は無視
        return "申し訳ありません、応答を生成できませんでした。" 

# Main Content Area (No login check needed)
if not supabase:
     st.error("Supabase is not configured. Please check your environment variables (SUPABASE_URL, SUPABASE_KEY).")
elif st.session_state.page == 'Diary':
    st.header("AI Chat Diary")
    st.subheader("日記を書く - AIとチャット")

    # --- Display "On This Day" Entries ---
    past_entries = get_on_this_day_entries()
    if past_entries:
        st.markdown("---")
        with st.container(border=True):
            st.subheader(f":calendar: 過去の同じ日...")
            for entry in past_entries:
                try:
                    # Attempt to parse ISO format with timezone offset
                    created_dt = datetime.fromisoformat(entry['created_at'].replace('Z', '+00:00'))
                    display_time = created_dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                     display_time = entry['created_at'] # Fallback to raw string
                summary = entry['summary'] if entry['summary'] else '(No summary)'
                expander_title = f"{display_time} - {summary}"
                with st.expander(expander_title):
                    st.markdown("**Full Text:**")
                    st.text(entry['body'] if entry['body'] else "") # Handle potential None
                    st.markdown("---**Analysis:**")
                    emotion_label = entry.get('emotion_label', "Unknown")
                    emotion_score = entry.get('emotion_score')
                    emotion_score_str = f" (Score: {emotion_score:.2f})" if emotion_score is not None else ""
                    st.write(f"検出された感情: {emotion_label}{emotion_score_str}")
        st.markdown("---")

    # --- Date/Time Selection ---
    st.markdown("**日記の日時を選択してください:**")
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("日付", value=date.today(), max_value=date.today(), key="entry_date", label_visibility="collapsed")
    with col2:
        # Get current time considering local timezone if possible, default to now()
        try:
            now_time = datetime.now().time()
        except Exception:
            now_time = time(12,0) # Fallback time
        selected_time = st.time_input("Time", value=now_time, key="entry_time", label_visibility="collapsed")

    # Combine selected date and time
    entry_dt = None
    try:
        entry_dt = datetime.combine(selected_date, selected_time)
        st.caption(f"選択された日時: {entry_dt.strftime('%Y-%m-%d %H:%M')}")
    except TypeError:
        st.error("選択された日付または時間が無効です。")
        entry_dt = None # Ensure it's None if combination fails
    st.markdown("---")


    # --- Chat History Display ---
    st.markdown("**AIアシスタントとチャット:**")
    chat_container = st.container(height=400)
    with chat_container:
         if not st.session_state.messages:
             initial_greeting = "こんにちは！今日はどんな一日でしたか？日記を書くお手伝いをしますね。"
             # Ensure initial greeting is added only once and has correct structure
             if not any(msg['role'] == 'assistant' and msg['content'] == initial_greeting for msg in st.session_state.messages):
                  st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
             # Display initial greeting immediately
             st.chat_message("assistant").markdown(initial_greeting)
         else:
             # Display all messages from session state
             for message in st.session_state.messages:
                 with st.chat_message(message["role"]):
                     st.markdown(message["content"])

    # --- Chat Input and Handling ---
    if prompt := st.chat_input("AIにメッセージを送る (例: 今日は〇〇がありました)"):
        # Append user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response or save action
        with st.spinner("AIが応答を考えています..."):
            # Pass a copy of messages to avoid potential modification issues
            response_data = get_ai_response(list(st.session_state.messages))

        # --- Handle Save Action or Normal Response ---
        if isinstance(response_data, dict) and response_data.get("action") == "save":
            # --- Perform Automated Save Operation ---
            diary_body_generated = response_data.get("body", "(日記の生成に失敗しました)")
            st.success("AIが会話を基に日記を作成しました。保存・分析します...")

            if not diary_body_generated.startswith("("): # Check if generation was successful
                if entry_dt is None:
                    st.error("日記の日時が無効なため、保存できませんでした。ページ上部で有効な日時を選択してください。")
                elif entry_dt > datetime.now():
                    st.error("未来の日時で日記を保存することはできません。")
                else:
                    # Proceed with saving the AI-generated diary
                    with st.spinner('AI生成日記を保存し、分析しています...'):
                        entry_id, entry_creation_time_iso = add_diary_entry(diary_body_generated, entry_datetime=entry_dt)

                        if entry_id and entry_creation_time_iso:
                            st.info(f"AI生成日記を保存しました (ID: {entry_id}). 感情と要約を生成中...")
                            analysis_success = True
                            short_summary = "(要約生成失敗)" # Placeholder
                            emotion_result = {"label": "Unknown", "score": None}

                            # Run emotion analysis on the generated diary body
                            if model:
                                emotion_result = analyze_emotion(diary_body_generated)
                                if not add_emotion_record(entry_id, entry_creation_time_iso, emotion_result['label'], emotion_result.get('score')):
                                    st.error("感情データの保存に失敗しました。")
                                    analysis_success = False

                                # Also generate short summary for list view
                                short_summary = generate_summary(diary_body_generated)
                                if not update_entry_summary(entry_id, short_summary):
                                     st.warning("一覧表示用の短い要約の保存に失敗しました。")
                                     # Don't necessarily mark analysis as failed for this

                            else:
                                st.warning("AIモデルが設定されていません。感情分析と要約生成をスキップします。")
                                update_entry_summary(entry_id, "(AI未設定)") # Indicate AI wasn't used
                                if not add_emotion_record(entry_id, entry_creation_time_iso, "Unknown", None):
                                    st.error("デフォルト感情データの保存に失敗しました。")
                                # Analysis didn't fully run
                                analysis_success = False

                            if analysis_success:
                                 st.success(f"AI生成日記を保存・分析しました！ Emotion: {emotion_result.get('label', 'N/A')}")
                            else:
                                 st.warning("AI生成日記は保存しましたが、分析中に問題が発生しました。")

                            st.session_state.messages = [] # Clear chat after save
                            py_time.sleep(2)
                            st.rerun()
                        else:
                            st.error("AI生成日記の保存に失敗しました。データベースエラーを確認してください。")
            else:
                 st.error("AIによる日記本文の生成に失敗しました。保存できませんでした。")

        else: # Normal AI chat response (or error message from get_ai_response)
            ai_response_text = response_data if isinstance(response_data, str) else "AIからの予期しない応答です。"
            with st.chat_message("assistant"):
                st.markdown(ai_response_text)
            # Append AI response to state only if it's not an error/action
            if isinstance(response_data, str):
                 st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
            # Rerun might cause issues with chat input losing focus, often better without it.
            # st.rerun()


    st.markdown("---")

    # --- Manual Save Button ---
    if st.button("[保存] 現在の会話から日記を作成"):
        user_messages = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        diary_body_combined = "\n".join(user_messages)

        if diary_body_combined:
            st.warning("保存を実行します。AIによる会話全体の要約は行われません。")
            if entry_dt is None:
                st.warning("日記の日時が無効なため、保存を中断しました。")
            elif entry_dt > datetime.now():
                st.error("未来の日時で日記を保存することはできません。")
            else:
                with st.spinner('日記を保存し、分析しています...'):
                    entry_id, entry_creation_time_iso = add_diary_entry(diary_body_combined, entry_datetime=entry_dt)
                    if entry_id and entry_creation_time_iso:
                        st.info(f"保存完了 (ID: {entry_id}). 分析中...")
                        analysis_success = True
                        summary_short = "(保存)"
                        emotion_result = {"label": "Unknown", "score": None}
                        if model:
                            summary_short = generate_summary(diary_body_combined)
                            update_entry_summary(entry_id, summary_short)
                            emotion_result = analyze_emotion(diary_body_combined)
                            if not add_emotion_record(entry_id, entry_creation_time_iso, emotion_result['label'], emotion_result.get('score')):
                                st.error("感情データの保存に失敗しました。")
                                analysis_success = False
                        else:
                            st.warning("AIモデル未構成。分析スキップ。")
                            update_entry_summary(entry_id, summary_short) # Save manual indicator
                            if not add_emotion_record(entry_id, entry_creation_time_iso, "Unknown", None):
                                 st.error("デフォルト感情データの保存失敗。")
                            analysis_success = False # Mark as not fully analyzed

                        if analysis_success:
                             st.success(f"保存・分析完了！ Summary: '{summary_short}', Emotion: {emotion_result.get('label', 'N/A')}")
                        else:
                             st.warning("保存は完了しましたが、分析で問題発生。")
                        st.session_state.messages = [] # Clear chat
                        py_time.sleep(1)
                        st.rerun()
                    else:
                        st.error("日記保存に失敗しました。")
        else: # No user messages to save
            st.warning("会話内容がありません。")

elif st.session_state.page == 'View':
    st.subheader("View Past Entries")
    with st.expander("Filters", expanded=False):
        with st.form(key='filter_form'):
            available_emotions = get_distinct_emotions() # Removed user_id
            filter_keyword = st.text_input("Keyword in Body", value=st.session_state.keyword_filter)
            col1, col2 = st.columns(2)
            with col1:
                filter_start_date = st.date_input("Start Date", value=st.session_state.start_date_filter)
            with col2:
                # Use today() if session state is None
                default_end = st.session_state.end_date_filter if st.session_state.end_date_filter else date.today()
                filter_end_date = st.date_input("End Date", value=default_end)
            filter_emotions = st.multiselect("Emotion Label", options=available_emotions, default=st.session_state.emotion_filter)
            submitted = st.form_submit_button("Apply Filters")
            if submitted:
                if filter_start_date and filter_end_date and filter_start_date > filter_end_date:
                    st.error("開始日は終了日よりも前でなければなりません。")
                else:
                    # Update session state with new filter values
                    st.session_state.keyword_filter = filter_keyword
                    st.session_state.start_date_filter = filter_start_date
                    st.session_state.end_date_filter = filter_end_date
                    st.session_state.emotion_filter = filter_emotions
                    st.session_state.filters_applied = True # Mark filters as active
                    st.rerun() # Rerun to apply filters

        if st.button("Clear Filters"):
            # Reset filter session state and rerun
            st.session_state.filters_applied = False
            st.session_state.keyword_filter = ""
            st.session_state.start_date_filter = None
            st.session_state.end_date_filter = None
            st.session_state.emotion_filter = []
            st.rerun()

    # Fetch entries based on current session state filters
    entries = get_filtered_entries(
                    keyword=st.session_state.keyword_filter,
                    start_date=st.session_state.start_date_filter,
                    end_date=st.session_state.end_date_filter,
                    selected_emotions=st.session_state.emotion_filter
                )

    if entries:
        entry_count = len(entries)
        st.write(f"条件に一致する{entry_count}件の日記を見つけました。" if st.session_state.filters_applied else f"{entry_count}件の日記を見つけました。")
        for entry in entries:
            entry_id_for_loop = entry['id'] # Use a distinct variable for the button key
            try:
                # Parse ISO timestamp for display
                created_dt = datetime.fromisoformat(entry['created_at'].replace('Z', '+00:00'))
                display_time = created_dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                display_time = entry['created_at'] # Fallback
            summary = entry.get('summary') if entry.get('summary') else '(No summary)'
            expander_title = f"{display_time} - {summary}"

            with st.expander(expander_title):
                st.markdown("**Full Text:**")
                st.text(entry.get('body', '')) # Handle potential None
                st.markdown("---**Analysis:**")
                emotion_label = entry.get('emotion_label', "Unknown")
                emotion_score = entry.get('emotion_score')
                emotion_score_str = f" (Score: {emotion_score:.2f})" if emotion_score is not None else ""
                st.write(f"検出された感情: {emotion_label}{emotion_score_str}")

                # --- Add Delete Button ---
                st.markdown("---") # Separator before button
                delete_button_key = f"delete_entry_{entry_id_for_loop}"
                if st.button("この日記を削除する", key=delete_button_key, type="primary"):
                    if delete_entry(entry_id_for_loop):
                        st.success(f"日記 (ID: {entry_id_for_loop}) を削除しました。")
                        # Give user time to see the message before rerun
                        py_time.sleep(1)
                        st.rerun() # Refresh the list
                    else:
                        # Error message is shown by delete_entry function
                        st.error(f"日記 (ID: {entry_id_for_loop}) の削除中にエラーが発生しました。")
                        # Keep the entry visible until successful deletion/refresh

    elif st.session_state.filters_applied:
        st.info("条件に一致する日記は見つかりませんでした。")
    else:
        st.info("まだ日記を書いていません。")

elif st.session_state.page == 'Visualize':
    st.subheader("時間経過に沿った感情の可視化")
    col1, col2 = st.columns(2)
    with col1:
        viz_start = st.date_input("開始日", value=st.session_state.viz_start_date, key="viz_start_date_picker")
    with col2:
        viz_end = st.date_input("End Date", value=st.session_state.viz_end_date, key="viz_end_date_picker")

    # Update session state if dates change (optional, could just use local vars)
    st.session_state.viz_start_date = viz_start
    st.session_state.viz_end_date = viz_end

    if viz_start and viz_end and viz_start > viz_end:
        st.error("開始日は終了日よりも前でなければなりません。")
    else:
        emotion_data = get_emotion_data(viz_start, viz_end) # Removed user_id

        if emotion_data:
            # Convert to DataFrame for plotting
            df = pd.DataFrame(emotion_data)

            # Ensure 'recorded_at' is datetime objects
            try:
                 df['recorded_at'] = pd.to_datetime(df['recorded_at'])
            except Exception as e:
                 st.error(f"Error converting 'recorded_at' to datetime: {e}")
                 st.dataframe(df) # Show raw data if conversion fails
                 df = pd.DataFrame() # Prevent further processing

            if not df.empty:
                df.sort_values(by='recorded_at', inplace=True)
                st.write(f"Displaying {len(df)} emotion records from {viz_start.strftime('%Y-%m-%d')} to {viz_end.strftime('%Y-%m-%d')}.\"")

                # --- Map emotions to numerical sentiment values (-1 to 1) --
                def map_emotion_to_sentiment(row):
                    label = row.get('emotion_label')
                    score = row.get('emotion_score') # Expected to be -1.0 to 1.0 from AI

                    # Prioritize using the AI score if available and valid
                    if score is not None:
                        try:
                            return float(score)
                        except (ValueError, TypeError):
                             pass # Fallback if score is not a valid float

                    # Fallback default sentiment values based on label
                    default_sentiments = {
                        'Positive': 0.7, 'Negative': -0.7, 'Neutral': 0.0,
                        'Joy': 0.8, 'Sadness': -0.6, 'Anger': -0.7,
                        'Surprise': 0.2, 'Fear': -0.5,
                        'Unknown': 0.0, 'Error': 0.0
                    }
                    return default_sentiments.get(label, 0.0) # Default to 0.0

                df['sentiment_value'] = df.apply(map_emotion_to_sentiment, axis=1)

                # --- Create the Line Plot --
                fig, ax = plt.subplots(figsize=(12, 6))
                if not df.empty: # Check again after potential conversion errors
                    ax.plot(df['recorded_at'], df['sentiment_value'], marker='o', linestyle='-', label='Sentiment Trend')

                    # --- Add Date Annotations to each point --
                    for index, row in df.iterrows():
                         # Format as MM-DD HH:MM
                        try:
                             datetime_str = row['recorded_at'].strftime('%m-%d %H:%M')
                             ax.text(row['recorded_at'], row['sentiment_value'] + 0.05, datetime_str,
                                    ha='center', va='bottom', fontsize=8, color='gray')
                        except Exception:
                             pass # Ignore annotation errors

                    ax.set_title('感情のタイムライン')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('感情スコア (-1 to 1)')
                    ax.set_ylim(-1.1, 1.1) # Consistent Y-axis
                    ax.legend()
                    plt.xticks(rotation=45)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                    ax.grid(True, axis='y', linestyle='--')
                    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Zero line
                    plt.tight_layout()
                    st.pyplot(fig)

                    with st.expander("感情値を含むデータを表示"):
                         st.dataframe(df[['recorded_at', 'emotion_label', 'emotion_score', 'sentiment_value']])
                else:
                    st.info("選択された期間の感情データは見つかりませんでした。")
        else: # No emotion data returned from DB initially
            st.info("選択された期間の感情データは記録されていません。")

elif st.session_state.page == 'Calendar':
    st.subheader("カレンダー表示")
    entry_dates = get_entry_dates() # Removed user_id
    calendar_events = []
    if entry_dates:
        for entry_date_str in entry_dates:
            # Basic validation for date string format
            try:
                datetime.strptime(entry_date_str, '%Y-%m-%d') # Check format
                calendar_events.append({
                    "title": "Diary Entry",
                    "start": entry_date_str,
                    "allDay": True, # Mark as all-day event on the calendar
                    #"url": f"/?page=View&date={entry_date_str}" # Optional: Link to view page filtered by date
                })
            except ValueError:
                print(f"カレンダーの無効な日付形式をスキップ: {entry_date_str}")

    calendar_options = {
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,timeGridDay", # View options
        },
        "initialView": "dayGridMonth", # Default view
        "selectable": True, # Allow selecting dates/times
        "events": calendar_events # Add events here directly
        # "eventClick": # JS code to handle event click (e.g., navigate) - Complex
    }
    st.write("Dates with diary entries are marked.")
    # Using streamlit-calendar component
    calendar_state = calendar(
        # events=calendar_events, # Pass events via options
        options=calendar_options,
        key="diary_calendar" # Unique key for the component
    )

    # calendar_state might contain information about clicked dates/events if configured
    # print(calendar_state) # For debugging what the component returns

    if not entry_dates:
         st.info("カレンダーに表示する日記は見つかりませんでした。") 