# streamlit_app.py
import re
import io
import os
import tempfile
from datetime import datetime
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
from dateutil import parser as dtparser

# NLP & images
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

# Ensure NLTK resource
nltk.download('vader_lexicon')

st.set_page_config(page_title="Funngro Clan AI Dashboard", layout="wide", page_icon="🐾")

# ---------- Theme / Branding ----------
DASHBOARD_NAME = st.sidebar.text_input("Dashboard name", value="Funngro Dashboard")
st.sidebar.markdown("### Theme")
st.sidebar.write("Theme: Dark + Neon (you chose Option C)")
theme_note = """
Background: near-black. Neon accents: green (#39FF14), cyan (#00E5FF), pink (#FF3B81).
Upload your logo in Settings to replace the placeholder.
"""
st.sidebar.info(theme_note)

# ---------- Helper functions ----------
def parse_whatsapp_txt(raw_text):
    """
    Parse WhatsApp exported .txt into a DataFrame with columns: timestamp, sender, message
    Compatible with common WhatsApp export formats. Handles multiline messages.
    """
    # Common WhatsApp line patterns (many variants exist)
    # Try day-first parsing by default.
    line_re = re.compile(
        r'^(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APMapm\.]{2,5})?)\s*-\s*([^:]+?):\s*(.*)$'
    )
    rows = []
    last = None
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = line_re.match(line)
        if m:
            datepart, timepart, sender, msg = m.groups()
            timestr = f"{datepart} {timepart}"
            # Try parsing; prefer dayfirst
            try:
                ts = dtparser.parse(timestr, dayfirst=True, fuzzy=True)
            except Exception:
                try:
                    ts = dtparser.parse(timestr, fuzzy=True)
                except Exception:
                    ts = None
            rows.append({"timestamp": ts, "sender": sender.strip(), "message": msg.strip()})
            last = rows[-1]
        else:
            # continuation line -> append to last message
            if last is not None:
                last["message"] = last["message"] + " " + line
            else:
                # orphan line — skip
                pass
    df = pd.DataFrame(rows)
    # Remove rows without timestamp
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df

def compute_member_counts(df):
    counts = df['sender'].value_counts().reset_index()
    counts.columns = ['sender', 'message_count']
    return counts

def sentiment_analysis(df):
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['message'].astype(str).apply(lambda t: sid.polarity_scores(t)['compound'])
    df['sentiment_cat'] = df['sentiment_score'].apply(lambda s: 'positive' if s>0.05 else ('negative' if s<-0.05 else 'neutral'))
    return df

# simple keyword sector mapping (tuneable)
SECTORS = {
    "design": ["design","logo","poster","canva","figma","photoshop","banner"],
    "projects": ["project","deadline","task","submit","progress","work","report"],
    "motivation": ["motivate","motivation","motivated","encourage","inspire","energize"],
    "queries": ["question","how","help","doubt","query","can anyone","please help"],
    "education": ["exam","study","syllabus","homework","class","school","tuition"],
    "logistics": ["time","place","when","where","meeting","schedule","join"]
}

def detect_sector(text):
    t = text.lower()
    for sector, kws in SECTORS.items():
        for kw in kws:
            if kw in t:
                return sector
    return "other"

def activity_score(df):
    # weekly metrics -> produce a simple activity score [0,1]
    if df.empty: return 0.0
    by_day = df.set_index('timestamp').resample('D').agg({'message':'count','sender':lambda s: s.nunique()})
    by_day.columns = ['msg_count','active_members']
    if by_day['msg_count'].max() == 0:
        return 0.0
    # Normalize and weight
    score = (by_day['msg_count'].mean()/by_day['msg_count'].max())*0.6 + (by_day['active_members'].mean()/(by_day['active_members'].max()+1))*0.4
    return float(score)

def generate_weekly_poster(summary, filename):
    """
    Simple PNG poster generator using Pillow. summary: dict with keys
    """
    W, H = 1200, 675
    bg = (11,15,26)  # near-black
    neon_green = (57,255,20)
    neon_cyan = (0,229,255)
    neon_pink = (255,59,129)
    im = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(im)

    # Load a default font included in many environments - fallback
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    draw.text((40,30), f"{summary.get('clan_name','Funngro Clan')}", font=font_title, fill=neon_cyan)
    draw.text((40,90), f"Weekly Summary", font=font_sub, fill=neon_green)

    # Stats box
    x0 = 40; y0 = 150
    draw.rectangle([x0,y0, W-40, y0+200], outline=neon_pink, width=3)
    draw.text((x0+20, y0+20), f"Total messages: {summary.get('total_messages',0)}", font=font_sub, fill="white")
    draw.text((x0+20, y0+60), f"Unique members: {summary.get('unique_members',0)}", font=font_sub, fill="white")
    draw.text((x0+20, y0+100), f"Avg sentiment: {summary.get('avg_sentiment',0):.3f}", font=font_sub, fill="white")
    draw.text((x0+20, y0+140), f"Activity score: {summary.get('activity_score',0):.2f}", font=font_sub, fill="white")

    # Top contributors
    tc_x = 700; tc_y = 160
    draw.text((tc_x, tc_y-30), "Top Contributors", font=font_sub, fill=neon_green)
    top = summary.get('top_contributors',[])
    for i, (name, cnt) in enumerate(top[:5]):
        draw.text((tc_x, tc_y + i*30), f"{i+1}. {name} — {cnt} msgs", font=font_small, fill="white")

    # Footer suggestion
    suggestion = summary.get('suggestion','Keep the group active — try a 30-min sprint tomorrow!')
    draw.text((40, 380), "AI Suggestion:", font=font_sub, fill=neon_pink)
    draw.text((40, 420), suggestion, font=font_small, fill="white")

    im.save(filename)
    return filename

# ---------- Streamlit UI ----------
st.markdown(f"<h1 style='color:#00E5FF;'>🐾 {DASHBOARD_NAME}</h1>", unsafe_allow_html=True)
st.markdown("Upload a WhatsApp exported chat file (.txt). The dashboard will parse it and show analytics.")

uploaded = st.file_uploader("Upload WhatsApp chat (.txt)", type=["txt"], accept_multiple_files=False)

if uploaded is not None:
    try:
        raw = uploaded.getvalue().decode('utf-8', errors='replace')
    except Exception:
        raw = uploaded.getvalue().decode('latin-1', errors='replace')

    with st.spinner("Parsing chat and running AI analysis..."):
        df = parse_whatsapp_txt(raw)
        if df.empty:
            st.error("No messages found in the uploaded file. Make sure you uploaded a valid WhatsApp export (.txt).")
        else:
            # Normalize columns
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['message'] = df['message'].astype(str)
            df['sender'] = df['sender'].astype(str)

            # Basic stats
            total_messages = len(df)
            unique_members = df['sender'].nunique()

            # Member counts
            members = compute_member_counts(df)

            # Sentiment
            df = sentiment_analysis(df)
            avg_sentiment = df['sentiment_score'].mean()

            # Sectors
            df['sector'] = df['message'].apply(detect_sector)
            sector_counts = df['sector'].value_counts().to_dict()

            # Activity score (simple)
            recent_df = df[df['timestamp'] >= (df['timestamp'].max() - pd.Timedelta(days=30))]
            act_score = activity_score(recent_df)

            # Top contributors
            top_contrib = members.head(10).values.tolist()

            # AI Summary (simple rule-based)
            negative_pct = (df['sentiment_cat']=='negative').mean()
            if act_score < 0.2:
                health = "Low"
            elif act_score < 0.45:
                health = "Moderate"
            else:
                health = "Active"

            if negative_pct > 0.2:
                suggestion = "Many messages show negative tone. Consider a check-in or encouragement post."
            elif sector_counts.get('design',0) > 10:
                suggestion = "Lots of design requests — schedule a small design workshop or ask for volunteers."
            else:
                suggestion = "Great! Keep up regular check-ins and micro-challenges to keep engagement high."

            # Display top metrics
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            col1.metric("Total messages", total_messages)
            col2.metric("Unique members", unique_members)
            col3.metric("Avg sentiment", f"{avg_sentiment:.3f}")
            col4.metric("Clan health", health)

            # Charts
            st.markdown("### Activity over time")
            by_day = df.set_index('timestamp').resample('D').agg({'message':'count'}).rename(columns={'message':'messages'})
            fig1, ax1 = plt.subplots(figsize=(10,3))
            ax1.plot(by_day.index, by_day['messages'])
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Messages")
            ax1.grid(True, linestyle=':', alpha=0.4)
            st.pyplot(fig1)

            st.markdown("### Sentiment distribution")
            sent_counts = df['sentiment_cat'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.bar(sent_counts.index, sent_counts.values)
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

            st.markdown("### Improvement sectors (top)")
            sc = pd.Series(sector_counts).sort_values(ascending=False)
            st.bar_chart(sc)

            st.markdown("### Top contributors")
            st.table(members.head(10).assign(rank=lambda d: range(1, len(d)+1))[['rank','sender','message_count']].set_index('rank'))

            st.markdown("### Sample negative messages (for review)")
            neg_sample = df[df['sentiment_cat']=='negative'].sort_values('sentiment_score').head(10)
            if not neg_sample.empty:
                st.table(neg_sample[['timestamp','sender','message','sentiment_score']].reset_index(drop=True))
            else:
                st.info("No strongly negative messages detected in the uploaded chat.")

            # Generate weekly poster
            poster_summary = {
                "clan_name": DASHBOARD_NAME,
                "total_messages": total_messages,
                "unique_members": unique_members,
                "avg_sentiment": avg_sentiment,
                "activity_score": act_score,
                "top_contributors": top_contrib,
                "suggestion": suggestion
            }
            tmpdir = tempfile.gettempdir()
            poster_path = os.path.join(tmpdir, f"funngro_poster_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
            generate_weekly_poster(poster_summary, poster_path)
            st.image(Image.open(poster_path), caption="Weekly Poster (download below)", use_column_width=True)
            with open(poster_path, "rb") as f:
                st.download_button("Download Weekly Poster (PNG)", data=f, file_name="funngro_weekly_poster.png", mime="image/png")

            # Save parsed CSV to allow downloads
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download parsed messages CSV", data=csv_bytes, file_name="funngro_parsed_messages.csv", mime="text/csv")
else:
    st.info("Upload a WhatsApp exported chat (.txt) to begin. The dashboard parses messages and shows analytics.")
