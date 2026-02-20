import streamlit as st
import joblib
import uuid
import datetime

from src.preprocessing import clean_text

st.set_page_config(
    page_title="Support Ticket Dashboard",
    layout="wide"
)


if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""

if "show_results" not in st.session_state:
    st.session_state.show_results = False

tfidf_model = joblib.load("model/category_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

semantic_model = joblib.load("model/semantic_model.pkl")
embedder = joblib.load("model/embedder.pkl")

def assign_priority(category):

    high = ["Hardware", "HR Support"]
    medium = ["Access", "Administrative rights"]

    if category in high:
        return "High"
    elif category in medium:
        return "Medium"
    else:
        return "Low"

def priority_badge(priority):

    colors = {
        "High": "#ff4b4b",
        "Medium": "#ffb703",
        "Low": "#2ecc71"
    }

    return f"""
    <span style="
        background-color:{colors[priority]};
        padding:6px 12px;
        border-radius:8px;
        color:white;
        font-weight:bold;">
        {priority}
    </span>
    """

def generate_ticket_id():

    year = datetime.datetime.now().year
    unique = str(uuid.uuid4())[:6].upper()

    return f"TCK-{year}-{unique}"


st.sidebar.title("System Panel")

st.sidebar.info("""
**Support Intelligence Engine**

• Hybrid ML Classification  
• Semantic Understanding  
• Automated Priority Assignment  
""")

st.sidebar.success("System Status: Operational")

st.title("Support Ticket Intelligence Dashboard")
st.caption("Automated ticket classification and routing system")

st.divider()


st.session_state.ticket_text = st.text_area(
    "Enter Support Ticket",
    value=st.session_state.ticket_text,
    height=120,
    placeholder="Describe the issue or request..."
)

colA, colB = st.columns(2)

analyze = colA.button("Analyze Ticket")
next_ticket = colB.button("Next Ticket")

if next_ticket:
    st.session_state.ticket_text = ""
    st.session_state.show_results = False
    st.rerun()

if analyze:

    if len(st.session_state.ticket_text.split()) < 3:
        st.warning("Please enter meaningful ticket information.")
        st.stop()

    st.session_state.show_results = True

    ticket_id = generate_ticket_id()

    cleaned = clean_text(st.session_state.ticket_text)

    vec_tfidf = vectorizer.transform([cleaned])
    pred_tfidf = tfidf_model.predict(vec_tfidf)[0]

    vec_semantic = embedder.encode([cleaned])
    pred_semantic = semantic_model.predict(vec_semantic)[0]

    priority = assign_priority(pred_semantic)

    agreement = pred_tfidf == pred_semantic

if st.session_state.show_results:

    st.success("Ticket classified successfully")

    st.markdown("### Ticket Summary")

    st.markdown(f"""
    <div style="
        background-color:#1c1f26;
        padding:20px;
        border-radius:10px;
        border:1px solid #2a2f3a;
        color:white;
    ">

    <b>Ticket ID:</b> {ticket_id}<br><br>

    <b>Recommended Category:</b> {pred_semantic}<br><br>

    <b>Priority Level:</b> {priority_badge(priority)}<br><br>

    <b>Validation Status:</b> {"High confidence classification" if agreement else "Requires review"}

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Model Validation")

    col1, col2 = st.columns(2)

    col1.metric("Statistical Model", pred_tfidf)
    col2.metric("Semantic Model", pred_semantic)

    if agreement:
        st.success("Models agree — classification validated")
    else:
        st.info("Models differ — manual review recommended")

    st.divider()

    st.subheader("Operational Summary")

    st.write(f"""
This ticket has been automatically categorized as **{pred_semantic}**
with a **{priority}** priority level.

The classification enables faster routing to the appropriate support team.
""")

    st.caption("Hybrid ML engine combining statistical and semantic analysis.")
