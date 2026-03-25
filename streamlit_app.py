
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, List
from pymongo import MongoClient
import torch, datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="ESG AI Advisor", page_icon="🌿", layout="wide")

MONGODB_URI = "mongodb+srv://ilamathidhandapani_db_user:Esg12345@cluster0.fsy0sir.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

for key in ["chat", "feedback"]:
    if key not in st.session_state:
        st.session_state[key] = []

def detect_pillar(text):
    t = text.lower()
    social_words = ["worker","health","social","pm2.5","labor","community",
                    "welfare","safety","employee","gender","female","reskill","medical","esi"]
    gov_words    = ["governance","policy","cbam","compliance","tax","regulation",
                    "audit","fame","pli","carbon credit","ccts","reporting"]
    score_s = sum(1 for w in social_words if w in t)
    score_g = sum(1 for w in gov_words if w in t)
    if score_g > score_s: return "G"
    if score_s > 0:       return "S"
    return "E"

PILLAR_LABEL = {"E": "ENVIRONMENT", "S": "SOCIAL", "G": "GOVERNANCE"}
PILLAR_COLOR = {"E": "green", "S": "blue", "G": "orange"}

@st.cache_resource
def load_rag():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed  = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    vs = Chroma(
        persist_directory="./esg_vectorstore",
        embedding_function=embed,
        collection_name="esg_sustainability"
    )
    tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map="auto")

    class FlanT5LLM(LLM):
        @property
        def _llm_type(self): return "flan-t5"
        def _call(self, prompt: str, stop=None, **kwargs) -> str:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = mdl.generate(**inputs, max_new_tokens=250, do_sample=False)
            return tok.decode(outputs[0], skip_special_tokens=True)

    llm       = FlanT5LLM()
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    PROMPT = PromptTemplate(
        template="""You are an ESG advisor for auto manufacturing in India.
Answer about ENVIRONMENT (CO2, energy, water), SOCIAL (worker health, welfare), GOVERNANCE (policy, CBAM).
Use context to give specific actionable advice with numbers.
Context: {context}
Question: {question}
Detailed Answer:""",
        input_variables=["context", "question"]
    )
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )
    return chain, retriever

@st.cache_resource
def get_db():
    try:
        client = MongoClient(MONGODB_URI)
        db = client["esg_project"]
        db.list_collection_names()
        return db
    except:
        return None

with st.sidebar:
    st.title("ESG AI Advisor")
    st.markdown("**Auto Manufacturing | India**")
    st.divider()
    st.markdown("### Crisis Metrics")
    st.metric("CO2 Emissions",   "1.82 t/cap",  "Rising")
    st.metric("PM2.5 Level",     "90.87 ug/m3", "18x WHO limit")
    st.metric("Coal Dependency", "75.31%",       "Must reduce")
    st.metric("Life Expectancy", "69.42 yrs",    "Below avg")
    st.divider()
    db = get_db()
    if db is not None:
        st.markdown("### MongoDB Stats")
        st.metric("Questions Asked", db.chat_history.count_documents({}))
        st.metric("Feedbacks",       db.feedback.count_documents({}))
        st.metric("Forecasts Run",   db.forecasts.count_documents({}))
        st.metric("Feed Records",    db.esg_feeds.count_documents({}))
    else:
        st.warning("MongoDB not connected")

st.title("ESG Decision Support System")
st.markdown("**RAG | Flan-T5 | ChromaDB | MongoDB | Auto Manufacturing India**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Forecast", "Dashboard", "Feedback", "Live Feed"])

with tab1:
    st.markdown("### Ask the ESG AI Advisor")
    suggestions = [
        "What welfare programs should we have for workers?",
        "How does PM2.5 affect workers and what are solutions?",
        "What community initiatives should our factory run?",
        "How do we reduce CO2 from our factory?",
        "What is CBAM and how does it affect us?",
        "How do we improve our ESG rating?"
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"q{i}", use_container_width=True):
                st.session_state.chat.append({"role": "user", "content": s})
                with st.spinner("Searching ChromaDB and generating answer..."):
                    chain, retriever = load_rag()
                    answer = chain.invoke(s)
                    docs   = retriever.invoke(s)
                pillar = detect_pillar(s)
                st.session_state.chat.append({
                    "role": "ai", "content": answer, "pillar": pillar,
                    "sources": [d.metadata["source"] for d in docs]
                })
                db = get_db()
                if db is not None:
                    db.chat_history.insert_one({
                        "question": s, "answer": answer, "pillar": pillar,
                        "timestamp": datetime.datetime.utcnow()
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
                st.markdown(f"**{PILLAR_LABEL[pillar]}**")
                st.write(msg["content"])
                if "sources" in msg:
                    with st.expander("Sources"):
                        for src in set(msg["sources"]):
                            st.caption(f"- {src}")
    user_q = st.chat_input("Ask anything about ESG...")
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        with st.spinner("Retrieving and generating..."):
            chain, retriever = load_rag()
            answer = chain.invoke(user_q)
            docs   = retriever.invoke(user_q)
        pillar = detect_pillar(user_q)
        st.session_state.chat.append({
            "role": "ai", "content": answer, "pillar": pillar,
            "sources": [d.metadata["source"] for d in docs]
        })
        db = get_db()
        if db is not None:
            db.chat_history.insert_one({
                "question": user_q, "answer": answer, "pillar": pillar,
                "timestamp": datetime.datetime.utcnow()
            })
        st.rerun()

with tab2:
    st.markdown("### ESG Indicator Forecast")
    try:
        df_esg = pd.read_csv("/content/ESGData.csv")
        india  = df_esg[df_esg["Country Name"] == "India"]
        indicators = india["Indicator Name"].tolist()
        col1, col2 = st.columns([2, 1])
        with col1:
            chosen = st.selectbox("Select ESG Indicator:", indicators)
        with col2:
            year = st.slider("Forecast to year:", 2021, 2050, 2030)
        if st.button("Run Forecast", type="primary"):
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
                fig.add_trace(go.Scatter(x=[a[0] for a in avail], y=[a[1] for a in avail],
                    mode="lines+markers", name="Historical", line=dict(color="blue", width=2)))
                fig.add_trace(go.Scatter(x=forecast_yrs, y=forecast_vals.flatten(),
                    mode="lines", name="Forecast", line=dict(color="red", width=2, dash="dash")))
                fig.update_layout(title=chosen, xaxis_title="Year", yaxis_title="Value", height=400)
                st.plotly_chart(fig, use_container_width=True)
                db = get_db()
                if db is not None:
                    db.forecasts.insert_one({
                        "indicator": chosen, "target_year": year,
                        "predicted": pred, "timestamp": datetime.datetime.utcnow()
                    })
                    st.success("Forecast saved to MongoDB!")
    except Exception as e:
        st.error(f"Error: {e}")

with tab3:
    st.markdown("### ESG Crisis Dashboard")
    crisis_data = {
        "Indicator": ["CO2 (t/cap)", "PM2.5 (ug/m3)", "Coal %", "Renewable %", "Life Expectancy", "Female Labor %"],
        "Current":   [1.82, 90.87, 75.31, 15.34, 69.42, 27.38],
        "Target":    [1.20,  5.00, 30.00, 50.00, 75.00, 45.00],
        "Status":    ["High Risk", "Critical", "High Risk", "Low", "Below Avg", "Low"]
    }
    df_crisis = pd.DataFrame(crisis_data)
    st.dataframe(df_crisis, use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_crisis, x="Indicator", y=["Current", "Target"],
                     barmode="group", title="Current vs Target",
                     color_discrete_map={"Current": "#ef4444", "Target": "#22c55e"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        action_data = {
            "Action": ["Rooftop Solar", "ZLD Water", "HEPA Filtration", "EV Transition", "ESI Health"],
            "Cost (Rs Lakh)": [300, 65, 20, 500, 10],
            "Impact Score":   [9, 8, 9, 10, 7]
        }
        fig2 = px.scatter(pd.DataFrame(action_data), x="Cost (Rs Lakh)", y="Impact Score",
                          text="Action", title="Cost vs Impact",
                          color="Impact Score", color_continuous_scale="RdYlGn")
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)
    db = get_db()
    if db is not None and db.chat_history.count_documents({}) > 0:
        st.markdown("### Recent Questions from MongoDB")
        for r in db.chat_history.find().sort("timestamp", -1).limit(5):
            pillar = r.get("pillar", "E")
            with st.expander(f"[{PILLAR_LABEL.get(pillar,'ESG')}] {str(r.get('question',''))[:80]}"):
                st.write(r.get("answer",""))
                st.caption(f"Time: {r.get('timestamp','')}")

with tab4:
    st.markdown("### Human Expert Feedback")
    col1, col2 = st.columns(2)
    with col1:
        acc    = st.slider("Accuracy (1-5):", 1, 5, 3)
        use    = st.slider("Usefulness (1-5):", 1, 5, 3)
        pillar = st.selectbox("ESG Pillar:", ["Social", "Environment", "Governance"])
    with col2:
        agr    = st.radio("Do you agree?", ["Yes", "Partially", "No"])
        domain = st.selectbox("Your domain:", ["Manufacturing", "HR / Social", "Environment",
                                               "Legal / Compliance", "Finance", "Research"])
    ctx = st.text_area("What did the AI miss?")
    if st.button("Submit Feedback", type="primary"):
        entry = {
            "accuracy": acc, "usefulness": use, "agreement": agr,
            "domain": domain, "pillar": pillar[0], "expert_context": ctx,
            "timestamp": datetime.datetime.utcnow()
        }
        db = get_db()
        if db is not None:
            db.feedback.insert_one(entry)
            st.success("Feedback saved to MongoDB!")
        else:
            st.success("Feedback recorded!")
    db = get_db()
    if db is not None and db.feedback.count_documents({}) > 0:
        st.divider()
        fb_list = list(db.feedback.find())
        fb_df   = pd.DataFrame(fb_list)
        if "accuracy" in fb_df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Accuracy",    f"{fb_df['accuracy'].mean():.1f}/5")
            c2.metric("Avg Usefulness",  f"{fb_df['usefulness'].mean():.1f}/5")
            c3.metric("Total Feedbacks", len(fb_df))

with tab5:
    st.markdown("### Live ESG Feed Data")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        feed_type    = st.selectbox("Feed Type:", ["news","alert","policy","sensor","report","user_input"])
        feed_title   = st.text_input("Title:")
        feed_content = st.text_area("Content:", height=100)
        feed_source  = st.text_input("Source:", value="manual")
        feed_tags    = st.text_input("Tags (comma-separated):")
        if st.button("Save Feed to MongoDB", type="primary"):
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
                    st.success("Feed saved to MongoDB!")
            else:
                st.warning("Please fill Title and Content")
    with col_b:
        filter_type  = st.selectbox("Filter:", ["all","news","alert","policy","sensor","report","user_input"])
        filter_limit = st.slider("Show last N:", 5, 50, 10)
    st.divider()
    db = get_db()
    if db is not None:
        query = {} if filter_type == "all" else {"feed_type": filter_type}
        feeds = list(db.esg_feeds.find(query).sort("timestamp", -1).limit(filter_limit))
        for f in feeds:
            with st.expander(f"[{f['feed_type'].upper()}] {f['title']}"):
                st.write(f["content"])
                st.caption(f"Source: {f.get('source','—')} | Tags: {', '.join(f.get('tags',[]))}")
    feed_q = st.text_input("Ask AI about feed data:")
    if feed_q:
        with st.spinner("Searching..."):
            chain, retriever = load_rag()
            answer = chain.invoke(feed_q)
        st.success(answer)
        db = get_db()
        if db is not None:
            db.chat_history.insert_one({
                "question": feed_q, "answer": answer,
                "pillar": detect_pillar(feed_q),
                "timestamp": datetime.datetime.utcnow()
            })
