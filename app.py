from dotenv import load_dotenv

load_dotenv()

# app.py
import os

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .env から環境変数（OPENAI_API_KEY）を読み込む
load_dotenv()


def get_llm_response(user_text: str, expert_choice: str) -> str:
    """
    入力テキスト(user_text)とラジオボタン選択値(expert_choice)を受け取り、
    LLMの回答を文字列で返す関数
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return "入力が空です。テキストを入力して送信してください。"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "OpenAI APIキーが設定されていません。\n"
            "ローカル実行の場合は .env に `OPENAI_API_KEY=...` を設定してください。\n"
            "Streamlit Community Cloud の場合は Secrets に `OPENAI_API_KEY` を設定してください。"
        )

    # ラジオ選択に応じてシステムメッセージ（専門家の振る舞い）を切り替え
    expert_system_messages = {
        "Python学習コーチ": (
            "あなたは熟練のPython学習コーチです。"
            "初心者にも分かるように、手順と理由を丁寧に説明し、必要なら短い例を示してください。"
            "不明点がある場合は、最初に確認質問を1〜2個だけしてから提案してください。"
        ),
        "プロダクト企画メンター": (
            "あなたは経験豊富なプロダクト企画メンターです。"
            "ユーザーの課題を整理し、仮説→検証→次アクションの順で具体的に提案してください。"
            "できれば箇条書きで、実行しやすい粒度にしてください。"
        ),
    }
    system_message = expert_system_messages.get(expert_choice, "あなたは有能なアシスタントです。")

    # Lesson8 で扱う形式に近い：ChatPromptTemplate + ChatOpenAI + invoke
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )

    chain = prompt | llm
    result = chain.invoke({"input": user_text})

    # result は AIMessage なので content を返す
    return result.content


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Streamlit LLMアプリ", page_icon="🤖", layout="centered")

st.title("🤖 Streamlit × LangChain LLMアプリ")
st.write(
    """
このアプリは **入力フォーム** に文章を入れて送信すると、**LangChain** を経由して **LLM** が回答を返します。  
左（下）の **ラジオボタン** で「どんな専門家として回答させるか」を切り替えられます。

**使い方**
1. 専門家タイプを選ぶ  
2. 入力欄に質問や相談を書いて送信  
3. 回答が下に表示されます
"""
)

expert_choice = st.radio(
    "専門家タイプを選択してください",
    ["Python学習コーチ", "プロダクト企画メンター"],
    horizontal=True,
)

user_text = st.text_input("入力フォーム（質問・相談内容）", placeholder="例：Pythonで辞書の使い方を教えて / 新機能の企画を整理したい")

send = st.button("送信", type="primary")

if send:
    with st.spinner("LLMに問い合わせ中..."):
        answer = get_llm_response(user_text, expert_choice)

    st.subheader("回答")
    st.write(answer)

st.divider()
st.caption(
    "※ローカル実行：.env に OPENAI_API_KEY を設定してください。"
    " ※デプロイ時：Streamlit Community Cloud の Secrets に OPENAI_API_KEY を設定してください。"
)
