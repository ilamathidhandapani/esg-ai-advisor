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

# ── API Keys ──────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["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"]
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
        st.error(f"Could not load ESGData.csv: {e}")
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

# ── Groq LLM ─────────────────────────────────────────────────
def get_llm_answer(question, context):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are an expert ESG sustainability advisor for an auto manufacturing company in India.

The company faces this interconnected crisis:
- 75% electricity from coal → CO2 rising (1.82 t/cap) → PM2.5 at 90.87 ug/m3 (18x WHO limit)
- Water scarcity → Worker health crisis (life expectancy 69.42 years)
- CBAM carbon tax on EU exports starts 2026

Answer questions about ALL THREE ESG pillars:
- ENVIRONMENT: CO2, energy, water, air quality, solar, ZLD
- SOCIAL: worker health, PM2.5 safety, welfare programs, community, gender inclusion
- GOVERNANCE: policy, CBAM, FAME III, CCTS, ESG rating, compliance

Use the retrieved context below to give specific, actionable advice with numbers.

Retrieved Context:
{context}

Question: {question}

Detailed Answer:"""

        response = client.chat.completions.create(
            model=llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {e}. Please check your Groq API key in Streamlit secrets."

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

PILLAR_LABEL = {'E': 'ENVIRONMENT', 'S': 'SOCIAL', 'G': 'GOVERNANCE'}
PILLAR_COLOR = {'E': 'green', 'S': 'blue', 'G': 'orange'}

# ── Session State ─────────────────────────────────────────────
for key in ["chat", "feedback"]:
    if key not in st.session_state:
        st.session_state[key] = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 ESG AI Advisor")
    st.markdown("**Auto Manufacturing | India**")
    st.divider()
    st.markdown("### 🔴 Crisis Metrics")
    st.metric("CO2 Emissions",   "1.82 t/cap",  "↑ Rising")
    st.metric("PM2.5 Level",     "90.87 µg/m³", "18x WHO limit")
    st.metric("Coal Dependency", "75.31%",       "↓ Must reduce")
    st.metric("Life Expectancy", "69.42 yrs",    "↓ Below avg")
    st.divider()
    db = get_db()
    if db is not None:
        st.markdown("### 📊 MongoDB Stats")
        st.metric("Questions Asked", db.chat_history.count_documents({}))
        st.metric("Feedbacks",       db.feedback.count_documents({}))
        st.metric("Forecasts Run",   db.forecasts.count_documents({}))
        st.metric("Feed Records",    db.esg_feeds.count_documents({}))
        st.success("✅ MongoDB Connected")
    else:
        st.warning("⚠️ MongoDB not connected")
    st.divider()
    st.markdown("### 🤖 AI Pipeline")
    st.info("RAG: ESG Docs → Retrieve → Groq LLaMA3 → Answer")

# ── Main ──────────────────────────────────────────────────────
st.title("🌿 ESG Decision Support System")
st.markdown("**RAG Pipeline | Groq LLaMA3 | MongoDB | Auto Manufacturing India**")

all_docs = build_knowledge_base()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI Chat", "📈 Forecast", "📊 Dashboard", "📝 Feedback", "📡 Live Feed"])

# ── TAB 1: AI CHAT ────────────────────────────────────────────
with tab1:
    st.markdown("### Ask the ESG AI Advisor")
    st.info("RAG Pipeline: Question → Retrieve from ESG Knowledge Base → Groq LLaMA3 → Answer")

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
                with st.spinner("🔍 Retrieving → 🤖 Groq LLaMA3 generating..."):
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

    st.divider()

    for msg in st.session_state.chat:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                pillar = msg.get("pillar", "E")
                color  = PILLAR_COLOR[pillar]
                st.markdown(f":{color}[**{PILLAR_LABEL[pillar]}**]")
                st.write(msg["content"])
                if "sources" in msg:
                    with st.expander("📚 Sources Retrieved by RAG"):
                        for src in set(msg["sources"]):
                            st.caption(f"• {src}")

    user_q = st.chat_input("Ask anything about ESG — environment, social, or governance...")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
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
    st.markdown("### 📈 ESG Indicator Forecast")
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
                    line=dict(color="blue", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_yrs, y=forecast_vals.flatten(),
                    mode="lines", name="Forecast",
                    line=dict(color="red", width=2, dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_yrs + forecast_yrs[::-1],
                    y=list(forecast_vals.flatten() + 1.96*std) +
                      list(forecast_vals.flatten() - 1.96*std)[::-1],
                    fill="toself", fillcolor="rgba(255,0,0,0.1)",
                    line=dict(color="rgba(255,255,255,0)"), name="95% CI"
                ))
                fig.update_layout(title=chosen, xaxis_title="Year",
                                  yaxis_title="Value", height=400)
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
    st.markdown("### 📊 ESG Crisis Dashboard — India Auto Manufacturing")

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
                     color_discrete_map={"Current": "#ef4444", "Target": "#22c55e"})
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
                          color="Impact Score", color_continuous_scale="RdYlGn")
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)

    db = get_db()
    if db is not None and db.chat_history.count_documents({}) > 0:
        st.markdown("### 💬 Recent Questions from MongoDB")
        recent = list(db.chat_history.find().sort("timestamp", -1).limit(5))
        for r in recent:
            pillar = r.get("pillar", "E")
            with st.expander(f"[{PILLAR_LABEL.get(pillar,'ESG')}] {str(r.get('question',''))[:80]}"):
                st.write(f"**Answer:** {r.get('answer','')}")
                st.caption(f"Time: {r.get('timestamp','')}")

# ── TAB 4: FEEDBACK ───────────────────────────────────────────
with tab4:
    st.markdown("### 📝 Human Expert Feedback")
    st.info("Your feedback is saved to MongoDB")

    col1, col2 = st.columns(2)
    with col1:
        acc    = st.slider("Accuracy (1-5):", 1, 5, 3)
        use    = st.slider("Usefulness (1-5):", 1, 5, 3)
        pillar = st.selectbox("ESG Pillar:", ["Social", "Environment", "Governance"])
    with col2:
        agr    = st.radio("Do you agree with the answer?", ["Yes", "Partially", "No"])
        domain = st.selectbox("Your domain:", ["Manufacturing", "HR / Social", "Environment",
                                               "Legal / Compliance", "Finance", "Research"])

    ctx = st.text_area("What did the AI miss? (add expert context):")
    alt = st.text_area("Suggest a better action or answer:")

    if st.button("Submit Feedback", type="primary"):
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
        st.divider()
        st.markdown("### 📊 Feedback Statistics")
        fb_list = list(db.feedback.find())
        fb_df   = pd.DataFrame(fb_list)
        if "accuracy" in fb_df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Accuracy",    f"{fb_df['accuracy'].mean():.1f}/5")
            c2.metric("Avg Usefulness",  f"{fb_df['usefulness'].mean():.1f}/5")
            c3.metric("Total Feedbacks", len(fb_df))

# ── TAB 5: LIVE FEED ──────────────────────────────────────────
with tab5:
    st.markdown("### 📡 Live ESG Feed Data")
    st.info("Save news, alerts, sensor data, policy updates → stored in MongoDB → queryable by AI")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("#### ➕ Add New Feed")
        feed_type    = st.selectbox("Feed Type:", ["news","alert","policy","sensor","report","user_input"])
        feed_title   = st.text_input("Title:")
        feed_content = st.text_area("Content:", height=100)
        feed_source  = st.text_input("Source:", value="manual")
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
                st.warning("Please fill Title and Content")

    with col_b:
        st.markdown("#### 🔍 Filter")
        filter_type  = st.selectbox("Type:", ["all","news","alert","policy","sensor","report","user_input"])
        filter_limit = st.slider("Show last N:", 5, 50, 10)

    st.divider()
    st.markdown("#### 📋 Stored Feeds")
    db = get_db()
    if db is not None:
        query = {} if filter_type == "all" else {"feed_type": filter_type}
        feeds = list(db.esg_feeds.find(query).sort("timestamp", -1).limit(filter_limit))
        ICONS = {"news":"🔵","alert":"🔴","policy":"🟢","sensor":"🟠","report":"🟣","user_input":"🟡"}
        for f in feeds:
            icon = ICONS.get(f["feed_type"], "⚪")
            with st.expander(f"{icon} [{f['feed_type'].upper()}] {f['title']} — {f['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f["content"])
                cols = st.columns(3)
                cols[0].caption(f"Source: {f.get('source','---')}")
                cols[1].caption(f"Tags: {', '.join(f.get('tags',[]))}")
                cols[2].caption(f"Indexed: {'✅' if f.get('indexed') else '⏳ Pending'}")

        st.divider()
        st.markdown("#### 🤖 Ask AI About Feed Data")
        feed_q = st.text_input("Ask a question about your stored feeds:")
        if feed_q:
            with st.spinner("🔍 Retrieving → 🤖 Generating..."):
                context, sources = retrieve_context(feed_q, all_docs)
                answer           = get_llm_answer(feed_q, context)
            st.success(answer)
            if db is not None:
                db.chat_history.insert_one({
                    "question": feed_q, "answer": answer,
                    "pillar": detect_pillar(feed_q),
                    "source": "feed_tab",
                    "timestamp": datetime.datetime.utcnow()
                })
    else:
        st.warning("MongoDB not connected")
