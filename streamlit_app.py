import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pymongo import MongoClient
import datetime
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq

st.set_page_config(page_title="ESG AI Advisor", page_icon="🌿", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%); }

.hero-header {
    background: linear-gradient(135deg, #1a3a6b 0%, #0d47a1 50%, #1565c0 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(13, 71, 161, 0.3);
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}

.hero-title {
    font-size: 2.5rem; font-weight: 700; color: white;
    margin: 0; letter-spacing: -0.5px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.hero-subtitle {
    font-size: 1rem; color: rgba(255,255,255,0.85);
    margin-top: 0.5rem; font-weight: 400; letter-spacing: 1px;
}

.hero-badges {
    display: flex; justify-content: center;
    gap: 10px; margin-top: 1rem; flex-wrap: wrap;
}

.badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white; padding: 4px 12px;
    border-radius: 20px; font-size: 0.75rem;
    font-weight: 500; backdrop-filter: blur(10px);
}

.metric-card {
    background: white; border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 16px rgba(13, 71, 161, 0.1);
    border-left: 4px solid #0d47a1;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(13, 71, 161, 0.2);
}

.metric-card.danger { border-left-color: #d32f2f; }
.metric-card.warning { border-left-color: #f57c00; }
.metric-card.success { border-left-color: #388e3c; }

.metric-label {
    font-size: 0.75rem; font-weight: 600;
    color: #546e7a; text-transform: uppercase; letter-spacing: 0.5px;
}

.metric-value { font-size: 1.8rem; font-weight: 700; color: #1a237e; margin: 0.2rem 0; }
.metric-delta { font-size: 0.8rem; font-weight: 500; color: #d32f2f; }

.chat-user {
    background: linear-gradient(135deg, #0d47a1, #1565c0);
    color: white; padding: 1rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0; max-width: 80%; margin-left: auto;
    box-shadow: 0 4px 12px rgba(13, 71, 161, 0.3);
    font-size: 0.95rem; line-height: 1.5;
}

.chat-ai {
    background: white; color: #1a237e;
    padding: 1rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 0; max-width: 85%;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border-left: 3px solid #0d47a1;
    font-size: 0.95rem; line-height: 1.6;
}

.pillar-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 12px; font-size: 0.7rem;
    font-weight: 700; letter-spacing: 0.5px; margin-bottom: 0.5rem;
}

.pillar-E { background: #e8f5e9; color: #2e7d32; }
.pillar-S { background: #e3f2fd; color: #1565c0; }
.pillar-G { background: #fff3e0; color: #e65100; }

.section-header {
    background: linear-gradient(135deg, #1a3a6b, #0d47a1);
    color: white; padding: 0.8rem 1.2rem;
    border-radius: 10px; font-size: 1.1rem;
    font-weight: 600; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}

.stButton > button {
    background: white !important;
    color: #0d47a1 !important;
    border: 2px solid #0d47a1 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
    padding: 0.6rem 1rem !important;
}

.stButton > button:hover {
    background: #0d47a1 !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(13, 71, 161, 0.3) !important;
}

.info-box {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border: 1px solid #90caf9; border-radius: 10px;
    padding: 0.8rem 1.2rem; color: #0d47a1;
    font-size: 0.9rem; margin-bottom: 1rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a6b 0%, #0d47a1 100%) !important;
}

section[data-testid="stSidebar"] * { color: white !important; }

section[data-testid="stSidebar"] .stMetric {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: white; border-radius: 10px;
    padding: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important; color: #546e7a !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0d47a1, #1565c0) !important;
    color: white !important;
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.loading-bar {
    height: 4px;
    background: linear-gradient(90deg, #0d47a1 25%, #42a5f5 50%, #0d47a1 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 2px; margin-bottom: 1rem;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f1f1; }
::-webkit-scrollbar-thumb { background: #0d47a1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── API Keys ──────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MONGODB_URI  = "mongodb+srv://ilamathidhandapani_db_user:Esg12345@cluster0.fsy0sir.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# ── MongoDB ───────────────────────────────────────────────────
@st.cache_resource
def get_db():
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db     = client["esg_project"]
        db.list_collection_names()
        return db
    except:
        return None

# ── Load ESG Data ─────────────────────────────────────────────
@st.cache_data
def load_esg_data():
    try:
        df    = pd.read_csv("ESGData.csv")
        india = df[df["Country Name"] == "India"].copy()
        return india
    except Exception as e:
        return pd.DataFrame()

# ── Build Knowledge Base ──────────────────────────────────────
@st.cache_data
def build_knowledge_base():
    india = load_esg_data()
    years = [str(y) for y in range(2000, 2021)]
    docs  = []

    if not india.empty:
        for _, row in india.iterrows():
            indicator  = row["Indicator Name"]
            avail_data = {y: row[y] for y in years if y in row.index and pd.notna(row[y])}
            if len(avail_data) < 5:
                continue
            values = list(avail_data.values())
            yrs    = list(avail_data.keys())
            latest = values[-1]
            oldest = values[0]
            change = latest - oldest
            trend  = "increasing" if change > 0 else "decreasing"
            text   = f"""ESG Indicator Report - India
Indicator: {indicator}
Time Period: {yrs[0]} to {yrs[-1]}
Latest Value ({yrs[-1]}): {latest:.2f}
Earliest Value ({yrs[0]}): {oldest:.2f}
Trend: {trend} by {abs(change):.2f} over {len(yrs)} years
Average annual change: {abs(change/len(yrs)):.3f} per year"""
            docs.append({"text": text, "source": f"ESGData_{indicator[:30]}", "indicator": indicator})

    sdg_docs = [
        {"text": """SDG 7: Affordable and Clean Energy
India: 95.24% electricity access but 75.31% from coal. Renewable only 15.34%.
Actions: Install rooftop solar 500kW-2MW. Cost Rs 2.5-3 crore/MW. ROI 5-7 years.
40% accelerated depreciation tax benefit. Join RE100. Purchase RECs.""",
         "source": "SDG_Framework", "indicator": "SDG7"},

        {"text": """SDG 13: Climate Action
India CO2: 1.82 metric tons per capita, rising since 2000.
India net zero target: 2070. 50% renewable by 2030 (Paris Agreement).
Risk: CBAM taxes high-carbon exports to EU from 2026.
Actions: Set SBTi targets, carbon offset programs, EV transition plan.""",
         "source": "SDG_Framework", "indicator": "SDG13"},

        {"text": """SDG 3: Good Health and Well-Being
India PM2.5: 90.87 micrograms/m3 - 18x WHO safe limit of 5.
Life expectancy: 69.42 years.
Actions: HEPA air filtration reduces PM2.5 by 85%. Provide N95 masks.
ESI health insurance for all workers. Annual lung function tests.
Legal risk: Liable under Occupational Safety Health and Working Conditions Code 2020.""",
         "source": "SDG_Framework", "indicator": "SDG3"},

        {"text": """SDG 6: Clean Water and Sanitation
Auto manufacturing uses 200-500 liters of water per vehicle.
Actions: Zero Liquid Discharge (ZLD) system. Cost Rs 50-80 lakh.
Recycles 95% of process water. Reduces freshwater intake by 70%.""",
         "source": "SDG_Framework", "indicator": "SDG6"},

        {"text": """SDG 8: Decent Work and Economic Growth
India unemployment: 7.11%. Female labor participation: only 27.38%.
EV transition risk: ICE engine workers may be displaced - reskilling essential.
Actions: Skill development for battery assembly, gender pay equity, safety audits.""",
         "source": "SDG_Framework", "indicator": "SDG8"},

        {"text": """Auto Manufacturing ESG Crisis Chain - India
Coal electricity 75.31% → CO2 rising 1.82 t/cap → PM2.5 at 90.87 ug/m3 (18x WHO)
→ Water scarcity → Worker health crisis → Low life expectancy 69.42 years
Solution: Solar energy → reduces CO2 → improves air → saves water → healthier workers.""",
         "source": "Domain_Knowledge", "indicator": "AutoMfgChain"},

        {"text": """Auto Manufacturing Immediate Actions - India
Action 1 - Rooftop Solar: 500kW-2MW. Cost Rs 2.5-3 cr/MW. ROI 5-7 years.
Action 2 - EV Transition: FAME III policy. PLI scheme Rs 25938 crore.
Action 3 - Water Recycling: ZLD system Rs 50-80 lakh. Recycles 95% process water.
Action 4 - Worker Health: HEPA filtration (85% PM2.5 reduction). ESI coverage.""",
         "source": "Domain_Knowledge", "indicator": "AutoMfgActions"},

        {"text": """India Government Policies for Sustainable Auto Sector
1. FAME III: Rs 25938 crore EV manufacturing incentive. Target 30% EV by 2030.
2. National Solar Mission: 500GW renewable by 2030. Net metering for rooftop solar.
3. CCTS Carbon Market: Earn tradeable carbon credits for emission reductions.
4. PLI Scheme: Production Linked Incentive for battery manufacturing.
5. Labour Code 2020: ESI for workers earning under Rs 21000 per month.""",
         "source": "Policy_Knowledge", "indicator": "IndiaPolicy"},

        {"text": """ESG Risk Assessment - Indian Auto Manufacturing
HIGH RISK: CO2 > 2.5 t/cap: EU CBAM risk from 2026.
PM2.5 > 35 ug: Worker health liability, OSHA violations.
Fossil fuel > 70%: Stranded asset risk.
ESG Rating: High risk → lower rating → higher cost of capital.
International buyers in EU and US require ESG compliance for procurement.""",
         "source": "Domain_Knowledge", "indicator": "RiskFramework"},

        {"text": """Worker Welfare Programs India Auto Manufacturing
1. ESI Health Insurance: Mandatory for workers earning < Rs 21000/month.
2. HEPA Air Filtration: Reduces PM2.5 by 85%. Cost Rs 5-10 lakh per unit.
3. Annual Health Checks: Lung function tests, blood tests for all shift workers.
4. Reskilling Programs: PMKVY scheme covers EV battery assembly training.
5. Female Worker Inclusion: Set 30% hiring target.
6. Mental Health Support: Partner with iCall or Vandrevala Foundation.""",
         "source": "Social_Knowledge", "indicator": "WorkerWelfare"},

        {"text": """Community Welfare Programs India Auto Manufacturing
1. Local hiring preference: Reduces commute emissions, supports local economy.
2. Skill development centers: Free ITI-level training for nearby village youth.
3. Water sharing: ZLD treated water shared with local farmers in dry season.
4. School support: Fund 1 government school per 500 factory workers.
5. Health camps: Monthly free medical camps for workers and families.
CSR mandate: Companies with net profit > Rs 5 crore must spend 2% on CSR.""",
         "source": "Social_Knowledge", "indicator": "CommunityWelfare"},
    ]

    return docs + sdg_docs

# ── RAG Retrieval ─────────────────────────────────────────────
def retrieve_context(question, docs, top_k=4):
    q_words = set(question.lower().split())
    scores  = []
    for doc in docs:
        doc_words = set(doc["text"].lower().split())
        score     = len(q_words & doc_words)
        scores.append((score, doc))
    scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = [d for _, d in scores[:top_k]]
    context  = "\n\n".join([d["text"] for d in top_docs])
    sources  = [d["source"] for d in top_docs]
    return context, sources

# ── Greeting / Small-talk Detection ──────────────────────────
GREETINGS = [
    "hi","hello","hey","hii","helo","howdy","sup","what's up","whats up",
    "good morning","good afternoon","good evening","good night",
    "how are you","how r u","how are u","how do you do","nice to meet you",
    "who are you","what are you","what can you do","help me","help",
    "thanks","thank you","thank u","ok","okay","great","awesome","cool",
    "bye","goodbye","see you","see ya","later"
]

def is_greeting(text: str) -> bool:
    t = text.strip().lower().rstrip("!?.")
    # exact match
    if t in GREETINGS:
        return True
    # short message (≤ 4 words) that starts with a greeting word
    words = t.split()
    if len(words) <= 4 and words[0] in GREETINGS:
        return True
    return False

def get_greeting_reply(text: str) -> str:
    t = text.strip().lower().rstrip("!?.")
    if any(w in t for w in ["bye","goodbye","see you","see ya","later"]):
        return "Goodbye! 👋 Feel free to come back anytime with your ESG questions."
    if any(w in t for w in ["thanks","thank you","thank u"]):
        return "You're welcome! 😊 Let me know if you have more ESG questions."
    if any(w in t for w in ["who are you","what are you","what can you do"]):
        return ("I'm your ESG AI Advisor 🌿 — I can help you with:\n"
                "• 🌍 Environment: CO2, solar, water, air quality\n"
                "• 👥 Social: worker health, welfare, gender inclusion\n"
                "• ⚖️ Governance: CBAM, FAME III, ESG ratings, compliance\n\n"
                "Go ahead, ask me anything!")
    if any(w in t for w in ["how are you","how r u","how are u","how do you do"]):
        return "I'm doing great, thanks for asking! 😊 How can I help you with ESG today?"
    # default greeting
    return ("Hello! 👋 I'm your ESG AI Advisor for Indian Auto Manufacturing.\n"
            "How can I help you today? You can ask me about:\n"
            "• CO2 & energy • Worker health • CBAM & policies")

# ── Groq LLM ─────────────────────────────────────────────────
def get_llm_answer(question, context):
    # Handle greetings & small talk without hitting RAG/LLM
    if is_greeting(question):
        return get_greeting_reply(question)

    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are a friendly ESG advisor for an Indian auto manufacturing company.

CRISIS CONTEXT:
- Coal 75% → CO2 1.82 t/cap → PM2.5 90.87 µg/m³ (18× WHO limit)
- Water scarcity → life expectancy 69.42 yrs
- CBAM carbon tax on EU exports from 2026

STRICT REPLY RULES:
1. Be conversational and friendly — like a knowledgeable colleague, not a report.
2. Keep answers SHORT: 3–5 bullet points max, each under 15 words.
3. Use bullet points (•) not paragraphs.
4. Lead with the most important point first.
5. Add 1 key number or cost if relevant.
6. End with ONE short follow-up offer like "Want details on any of these?"
7. Never write long paragraphs. Never repeat the question.

Retrieved Context:
{context}

Question: {question}

Short Answer:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Error: {e}"

# ── Pillar Detection ──────────────────────────────────────────
def detect_pillar(text):
    t = text.lower()
    social_words = ['worker','health','social','pm2.5','labor','labour','community',
                    'welfare','safety','employee','gender','female','reskill','medical','esi']
    gov_words    = ['governance','policy','cbam','compliance','tax','regulation',
                    'audit','esg rating','fame','pli','carbon credit','ccts','reporting']
    score_s = sum(1 for w in social_words if w in t)
    score_g = sum(1 for w in gov_words if w in t)
    if score_g > score_s: return 'G'
    if score_s > 0:       return 'S'
    return 'E'

PILLAR_LABEL = {'E': '🌍 ENVIRONMENT', 'S': '👥 SOCIAL', 'G': '⚖️ GOVERNANCE'}
PILLAR_CLASS = {'E': 'pillar-E', 'S': 'pillar-S', 'G': 'pillar-G'}

# ── Session State ─────────────────────────────────────────────
for key in ["chat", "feedback"]:
    if key not in st.session_state:
        st.session_state[key] = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 ESG AI Advisor")
    st.markdown("**Auto Manufacturing | India**")
    st.markdown("---")
    st.markdown("### 🔴 Crisis Metrics")
    st.metric("CO2 Emissions",   "1.82 t/cap",  "↑ Rising")
    st.metric("PM2.5 Level",     "90.87 µg/m³", "18x WHO limit")
    st.metric("Coal Dependency", "75.31%",       "↓ Must reduce")
    st.metric("Life Expectancy", "69.42 yrs",    "↓ Below avg")
    st.markdown("---")
    db = get_db()
    if db is not None:
        st.markdown("### 📊 MongoDB Stats")
        st.metric("💬 Questions", db.chat_history.count_documents({}))
        st.metric("📝 Feedbacks", db.feedback.count_documents({}))
        st.metric("📈 Forecasts", db.forecasts.count_documents({}))
        st.metric("📡 Feeds",     db.esg_feeds.count_documents({}))
        st.success("✅ MongoDB Connected")
    else:
        st.warning("⚠️ MongoDB not connected")
    st.markdown("---")
    st.markdown("### 🤖 AI Pipeline")
    st.markdown("""
    ```
    Question
       ↓
    RAG Retrieve
       ↓
    Groq LLaMA3
       ↓
    Answer + MongoDB
    ```
    """)

# ── Hero Header ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🌿 ESG Decision Support System</div>
    <div class="hero-subtitle">AI-Powered Sustainability Advisor for Indian Auto Manufacturing</div>
    <div class="hero-badges">
        <span class="badge">🤖 Groq LLaMA3</span>
        <span class="badge">📚 RAG Pipeline</span>
        <span class="badge">🍃 MongoDB Atlas</span>
        <span class="badge">🌍 ESG Analytics</span>
        <span class="badge">🏭 Auto Manufacturing</span>
    </div>
</div>
""", unsafe_allow_html=True)

all_docs = build_knowledge_base()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 AI Chat", "📈 Forecast", "📊 Dashboard", "📝 Feedback", "📡 Live Feed"
])

# ── TAB 1: AI CHAT ────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">💬 Ask the ESG AI Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">🔍 RAG Pipeline: Question → Retrieve ESG Knowledge → Groq LLaMA3 → Detailed Answer</div>', unsafe_allow_html=True)

    suggestions = [
        "What welfare programs should we have for workers?",
        "How does PM2.5 affect workers and what are the solutions?",
        "What community initiatives should our factory run?",
        "How do we reduce CO2 from our factory?",
        "What is CBAM and how does it affect us?",
        "How do we improve our ESG rating?",
    ]

    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"q{i}", use_container_width=True):
                st.session_state.chat.append({"role": "user", "content": s})
                if is_greeting(s):
                    answer  = get_greeting_reply(s)
                    sources = []
                else:
                    with st.spinner("🔍 Retrieving ESG knowledge → 🤖 Generating answer..."):
                        context, sources = retrieve_context(s, all_docs)
                        answer           = get_llm_answer(s, context)
                pillar = detect_pillar(s)
                st.session_state.chat.append({
                    "role": "ai", "content": answer,
                    "pillar": pillar, "sources": sources
                })
                db = get_db()
                if db is not None:
                    db.chat_history.insert_one({
                        "question": s, "answer": answer, "pillar": pillar,
                        "sources": sources, "timestamp": datetime.datetime.utcnow()
                    })
                st.rerun()

    st.markdown("---")

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            pillar  = msg.get("pillar", "E")
            pclass  = PILLAR_CLASS[pillar]
            plabel  = PILLAR_LABEL[pillar]
            sources = msg.get("sources", [])
            src_str = " • ".join(set(sources)) if sources else ""
            st.markdown(f"""
            <div class="chat-ai">
                <span class="pillar-badge {pclass}">{plabel}</span><br>
                🤖 {msg["content"]}
                <br><small style="color:#90a4ae; margin-top:8px; display:block;">📚 Sources: {src_str}</small>
            </div>
            """, unsafe_allow_html=True)

    user_q = st.chat_input("Say hi, or ask anything about ESG — environment, social, or governance...")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        if is_greeting(user_q):
            answer  = get_greeting_reply(user_q)
            sources = []
        else:
            with st.spinner("🔍 Retrieving → 🤖 Generating..."):
                context, sources = retrieve_context(user_q, all_docs)
                answer           = get_llm_answer(user_q, context)
        pillar = detect_pillar(user_q)
        st.session_state.chat.append({
            "role": "ai", "content": answer,
            "pillar": pillar, "sources": sources
        })
        db = get_db()
        if db is not None:
            db.chat_history.insert_one({
                "question": user_q, "answer": answer, "pillar": pillar,
                "sources": sources, "timestamp": datetime.datetime.utcnow()
            })
        st.rerun()

# ── TAB 2: FORECAST ───────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📈 ESG Indicator Forecast</div>', unsafe_allow_html=True)
    india = load_esg_data()

    if not india.empty:
        indicators = india["Indicator Name"].tolist()
        col1, col2 = st.columns([2, 1])
        with col1:
            chosen = st.selectbox("Select ESG Indicator:", indicators)
        with col2:
            year = st.slider("Forecast to year:", 2021, 2050, 2030)

        if st.button("🔮 Run Forecast", type="primary"):
            row   = india[india["Indicator Name"] == chosen].iloc[0]
            yrs_  = [str(y) for y in range(2000, 2021)]
            avail = [(int(y), float(row[y])) for y in yrs_ if y in row.index and pd.notna(row[y])]

            if len(avail) > 3:
                X = np.array([a[0] for a in avail]).reshape(-1, 1)
                Y = np.array([a[1] for a in avail])
                m = LinearRegression().fit(X, Y)
                pred = m.predict([[year]])[0]
                std  = (Y - m.predict(X)).std()

                forecast_yrs  = list(range(2021, year + 1))
                forecast_vals = m.predict(np.array(forecast_yrs).reshape(-1, 1))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Latest (2020)",     f"{Y[-1]:.2f}")
                c2.metric(f"Predicted {year}", f"{pred:.2f}", f"{pred-Y[-1]:+.2f}")
                c3.metric("Lower 95%",         f"{pred-1.96*std:.2f}")
                c4.metric("Upper 95%",         f"{pred+1.96*std:.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[a[0] for a in avail], y=[a[1] for a in avail],
                    mode="lines+markers", name="Historical",
                    line=dict(color="#0d47a1", width=3),
                    marker=dict(size=6, color="#0d47a1")
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_yrs, y=forecast_vals.flatten(),
                    mode="lines", name="Forecast",
                    line=dict(color="#f57c00", width=3, dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_yrs + forecast_yrs[::-1],
                    y=list(forecast_vals.flatten() + 1.96*std) +
                      list(forecast_vals.flatten() - 1.96*std)[::-1],
                    fill="toself", fillcolor="rgba(245,124,0,0.1)",
                    line=dict(color="rgba(255,255,255,0)"), name="95% CI"
                ))
                fig.update_layout(
                    title=dict(text=chosen, font=dict(size=14, color="#1a237e")),
                    xaxis_title="Year", yaxis_title="Value", height=420,
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter"),
                    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e0e0", borderwidth=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                db = get_db()
                if db is not None:
                    db.forecasts.insert_one({
                        "indicator": chosen, "target_year": year,
                        "predicted": float(pred),
                        "timestamp": datetime.datetime.utcnow()
                    })
                    st.success("💾 Forecast saved to MongoDB!")
            else:
                st.warning("Not enough data for this indicator.")
    else:
        st.error("ESGData.csv not found.")

# ── TAB 3: DASHBOARD ──────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📊 ESG Crisis Dashboard — India Auto Manufacturing</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card danger">
            <div class="metric-label">CO2 Emissions</div>
            <div class="metric-value">1.82</div>
            <div class="metric-delta">↑ t/cap — Rising</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card danger">
            <div class="metric-label">PM2.5 Level</div>
            <div class="metric-value">90.87</div>
            <div class="metric-delta">↑ µg/m³ — 18x WHO</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card warning">
            <div class="metric-label">Coal Dependency</div>
            <div class="metric-value">75.31%</div>
            <div class="metric-delta">↓ Must Reduce</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card warning">
            <div class="metric-label">Life Expectancy</div>
            <div class="metric-value">69.42</div>
            <div class="metric-delta">↓ yrs — Below Avg</div>
        </div>""", unsafe_allow_html=True)

    crisis_data = {
        "Indicator": ["CO2 (t/cap)", "PM2.5 (µg/m³)", "Coal %", "Renewable %", "Life Expectancy", "Female Labor %"],
        "Current":   [1.82, 90.87, 75.31, 15.34, 69.42, 27.38],
        "Target":    [1.20,  5.00, 30.00, 50.00, 75.00, 45.00],
        "Status":    ["🔴 High Risk", "🔴 Critical", "🔴 High Risk", "🟡 Low", "🟡 Below Avg", "🔴 Low"]
    }
    df_crisis = pd.DataFrame(crisis_data)
    st.dataframe(df_crisis, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_crisis, x="Indicator", y=["Current", "Target"],
                     barmode="group", title="Current vs Target ESG Values",
                     color_discrete_map={"Current": "#d32f2f", "Target": "#388e3c"})
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font=dict(family="Inter"), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        action_data = {
            "Action": ["Rooftop Solar", "ZLD Water", "HEPA Filtration", "EV Transition", "ESI Health"],
            "Cost (Rs Lakh)": [300, 65, 20, 500, 10],
            "Impact Score":   [9, 8, 9, 10, 7]
        }
        fig2 = px.scatter(pd.DataFrame(action_data),
                          x="Cost (Rs Lakh)", y="Impact Score",
                          text="Action", title="Action: Cost vs Impact",
                          color="Impact Score", color_continuous_scale="Blues",
                          size="Impact Score", size_max=30)
        fig2.update_traces(textposition="top center")
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           font=dict(family="Inter"), height=380)
        st.plotly_chart(fig2, use_container_width=True)

    db = get_db()
    if db is not None and db.chat_history.count_documents({}) > 0:
        st.markdown('<div class="section-header">💬 Recent Questions from MongoDB</div>', unsafe_allow_html=True)
        recent = list(db.chat_history.find().sort("timestamp", -1).limit(5))
        for r in recent:
            pillar = r.get("pillar", "E")
            with st.expander(f"{PILLAR_LABEL.get(pillar,'ESG')} — {str(r.get('question',''))[:80]}"):
                st.write(f"**Answer:** {r.get('answer','')}")
                st.caption(f"🕐 {r.get('timestamp','')}")

# ── TAB 4: FEEDBACK ───────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📝 Human Expert Feedback</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Your feedback is saved to MongoDB and helps improve the AI system</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        acc    = st.slider("⭐ Accuracy (1-5):", 1, 5, 3)
        use    = st.slider("💡 Usefulness (1-5):", 1, 5, 3)
        pillar = st.selectbox("🏷️ ESG Pillar:", ["Social", "Environment", "Governance"])
    with col2:
        agr    = st.radio("✅ Do you agree with the answer?", ["Yes", "Partially", "No"])
        domain = st.selectbox("🏢 Your domain:", ["Manufacturing", "HR / Social", "Environment",
                                                   "Legal / Compliance", "Finance", "Research"])

    ctx = st.text_area("❓ What did the AI miss? (add expert context):")
    alt = st.text_area("💭 Suggest a better action or answer:")

    if st.button("📤 Submit Feedback", type="primary"):
        entry = {
            "accuracy": acc, "usefulness": use,
            "agreement": agr, "domain": domain,
            "pillar": pillar[0],
            "expert_context": ctx, "alternative": alt,
            "timestamp": datetime.datetime.utcnow()
        }
        st.session_state.feedback.append(entry)
        db = get_db()
        if db is not None:
            db.feedback.insert_one(entry)
            st.success("✅ Feedback saved to MongoDB!")
        else:
            st.success("✅ Feedback recorded locally!")

    db = get_db()
    if db is not None and db.feedback.count_documents({}) > 0:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Feedback Statistics</div>', unsafe_allow_html=True)
        fb_list = list(db.feedback.find())
        fb_df   = pd.DataFrame(fb_list)
        if "accuracy" in fb_df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("⭐ Avg Accuracy",    f"{fb_df['accuracy'].mean():.1f}/5")
            c2.metric("💡 Avg Usefulness",  f"{fb_df['usefulness'].mean():.1f}/5")
            c3.metric("📝 Total Feedbacks", len(fb_df))

# ── TAB 5: LIVE FEED ──────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">📡 Live ESG Feed Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📥 Save news, alerts, sensor data, policy updates → stored in MongoDB → queryable by AI</div>', unsafe_allow_html=True)

    st.markdown("#### ➕ Add New Feed")
    col_a, col_b = st.columns(2)
    with col_a:
        feed_type    = st.selectbox("Feed Type:", ["news","alert","policy","sensor","report","user_input"])
        feed_title   = st.text_input("Title:")
        feed_source  = st.text_input("Source:", value="manual")
    with col_b:
        feed_content = st.text_area("Content:", height=120)
        feed_tags    = st.text_input("Tags (comma-separated):", placeholder="CO2, alert, social")

    if st.button("💾 Save Feed to MongoDB", type="primary"):
        if feed_title and feed_content:
            tags = [t.strip() for t in feed_tags.split(",") if t.strip()]
            db   = get_db()
            if db is not None:
                db.esg_feeds.insert_one({
                    "feed_type": feed_type, "title": feed_title,
                    "content": feed_content, "source": feed_source,
                    "tags": tags, "timestamp": datetime.datetime.utcnow(),
                    "indexed": False
                })
                st.success("✅ Feed saved to MongoDB!")
            else:
                st.error("❌ MongoDB not connected")
        else:
            st.warning("⚠️ Please fill Title and Content")

    st.markdown("---")
    st.markdown("#### 🤖 Ask AI About Feed Data")
    feed_q = st.text_input("Ask a question about your stored feeds:")
    if feed_q:
        if is_greeting(feed_q):
            answer = get_greeting_reply(feed_q)
        else:
            with st.spinner("🔍 Retrieving → 🤖 Generating..."):
                context, sources = retrieve_context(feed_q, all_docs)
                answer           = get_llm_answer(feed_q, context)
        st.markdown(f'<div class="chat-ai">🤖 {answer}</div>', unsafe_allow_html=True)
        db = get_db()
        if db is not None:
            db.chat_history.insert_one({
                "question": feed_q, "answer": answer,
                "pillar": detect_pillar(feed_q),
                "source": "feed_tab",
                "timestamp": datetime.datetime.utcnow()
            })
