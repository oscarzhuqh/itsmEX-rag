import streamlit as st
import pandas as pd

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# 2. STREAMLIT PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="ITSM Intelligent Assistant",
    page_icon="🎫",
    layout="wide"
)

st.title("🎫 ITSM Intelligent Assistant")

st.write(
    "Analyse ITSM ticket statistics and ask questions "
    "using traditional data analysis or RAG."
)


# ==================================================
# 3. LOAD DATASET
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/ITSM_Dataset.csv")


df = load_data()


# ==================================================
# 4. LOAD EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


embedding_model = load_embedding_model()


# ==================================================
# 5. CONNECT TO CHROMADB
# ==================================================

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model,
        collection_name="itsm_tickets"
    )


vectorstore = load_vectorstore()


# ==================================================
# 6. LOAD LLM
# ==================================================

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )


llm = load_llm()


# ==================================================
# 7. DATASET INFORMATION
# ==================================================

st.subheader("📁 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total ITSM Tickets",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Number of Columns",
        len(df.columns)
    )


# ==================================================
# 8. PRIORITY ANALYTICS DASHBOARD
# ==================================================

st.divider()

st.subheader("📊 ITSM Priority Analytics")

st.write(
    "Explore the distribution of ticket priority levels "
    "across the complete ITSM dataset."
)

show_dashboard = st.toggle(
    "Show Priority Dashboard"
)


if show_dashboard:

    # ----------------------------------------------
    # Calculate priority statistics
    # ----------------------------------------------

    priority_counts = (
        df["Priority"]
        .value_counts()
        .reindex(
            [
                "Critical",
                "High",
                "Medium",
                "Low"
            ],
            fill_value=0
        )
    )

    total_tickets = len(df)

    priority_percentages = (
        priority_counts
        / total_tickets
        * 100
    )


    # ----------------------------------------------
    # KPI CARDS
    # ----------------------------------------------

    st.subheader("Priority Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🔴 Critical",
            f"{priority_counts['Critical']:,}"
        )

    with col2:
        st.metric(
            "🟠 High",
            f"{priority_counts['High']:,}"
        )

    with col3:
        st.metric(
            "🟡 Medium",
            f"{priority_counts['Medium']:,}"
        )

    with col4:
        st.metric(
            "🟢 Low",
            f"{priority_counts['Low']:,}"
        )


    # ----------------------------------------------
    # PRIORITY DISTRIBUTION CHART
    # ----------------------------------------------

    st.subheader("Priority Distribution")

    priority_chart = pd.DataFrame({
        "Priority": priority_counts.index,
        "Number of Tickets": priority_counts.values
    })

    st.bar_chart(
        priority_chart,
        x="Priority",
        y="Number of Tickets"
    )


    # ----------------------------------------------
    # PRIORITY STATISTICS TABLE
    # ----------------------------------------------

    st.subheader("Priority Statistics")

    priority_statistics = pd.DataFrame({
        "Priority": priority_counts.index,
        "Tickets": priority_counts.values,
        "Percentage (%)": priority_percentages.values
    })

    priority_statistics["Percentage (%)"] = (
        priority_statistics["Percentage (%)"]
        .round(2)
    )

    st.dataframe(
        priority_statistics,
        width="stretch",
        hide_index=True
    )


    # ----------------------------------------------
    # HIGH-PRIORITY WORKLOAD
    # ----------------------------------------------

    urgent_tickets = (
        priority_counts["Critical"]
        + priority_counts["High"]
    )

    urgent_percentage = (
        urgent_tickets
        / total_tickets
        * 100
    )

    st.subheader("⚠️ High-Priority Workload")

    st.metric(
        "Critical + High Tickets",
        f"{urgent_tickets:,}",
        f"{urgent_percentage:.2f}% of all tickets"
    )


# ==================================================
# 9. QUERY ROUTER
# ==================================================

def determine_route(question):

    question_lower = question.lower()

    analytical_words = [
        "how many",
        "count",
        "total",
        "most",
        "least",
        "number of"
    ]

    for word in analytical_words:

        if word in question_lower:
            return "PANDAS"

    return "RAG"


# ==================================================
# 10. PANDAS ANALYTICAL ENGINE
# ==================================================

def pandas_answer(question):

    question_lower = question.lower()


    # ----------------------------------------------
    # PRIORITY QUESTIONS
    # ----------------------------------------------

    priorities = [
        "critical",
        "high",
        "medium",
        "low"
    ]

    for priority in priorities:

        if priority in question_lower:

            count = (
                df["Priority"]
                .astype(str)
                .str.lower()
                .eq(priority)
                .sum()
            )

            return (
                f"There are **{count:,} "
                f"{priority.title()} priority tickets** "
                f"in the complete dataset."
            )


    # ----------------------------------------------
    # STATUS QUESTIONS
    # ----------------------------------------------

    statuses = [
        "open",
        "closed",
        "resolved",
        "new",
        "in progress"
    ]

    for status in statuses:

        if status in question_lower:

            count = (
                df["Status"]
                .astype(str)
                .str.lower()
                .eq(status)
                .sum()
            )

            return (
                f"There are **{count:,} tickets "
                f"with status {status.title()}**."
            )


    # ----------------------------------------------
    # TOTAL TICKETS
    # ----------------------------------------------

    if (
        "total" in question_lower
        or "how many tickets" in question_lower
    ):

        return (
            f"The dataset contains "
            f"**{len(df):,} tickets**."
        )


    # ----------------------------------------------
    # UNSUPPORTED ANALYTICAL QUESTION
    # ----------------------------------------------

    return (
        "This appears to be an analytical question, "
        "but the current Pandas engine does not yet "
        "support this calculation."
    )


# ==================================================
# 11. RAG ENGINE
# ==================================================

def rag_answer(question):

    # ----------------------------------------------
    # Retrieve similar tickets
    # ----------------------------------------------

    results = vectorstore.similarity_search(
        question,
        k=5
    )


    if not results:

        return (
            "No relevant tickets were retrieved.",
            []
        )


    # ----------------------------------------------
    # Build context
    # ----------------------------------------------

    context = "\n\n".join(
        result.page_content
        for result in results
    )


    # ----------------------------------------------
    # RAG prompt
    # ----------------------------------------------

    prompt = f"""
You are an ITSM support assistant.

Answer the user's question using ONLY the ITSM ticket
records contained in the context below.

Rules:

1. Do not invent ticket information.

2. Only make statements supported by the retrieved
   ticket records.

3. When referring to a ticket, include its Ticket ID.

4. If multiple relevant tickets are available,
   mention them individually.

5. If the retrieved records do not contain enough
   information to answer the question, clearly state
   that there is insufficient information.

6. Do not claim that the retrieved records represent
   the entire ITSM database. They are only records
   retrieved by the RAG system.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


    # ----------------------------------------------
    # Send context to LLM
    # ----------------------------------------------

    response = llm.invoke(prompt)

    return response.content, results


# ==================================================
# 12. ITSM ASSISTANT
# ==================================================

st.divider()

st.subheader("🤖 Ask the ITSM Assistant")

st.write(
    "The Query Router automatically decides whether "
    "to use Pandas analysis or RAG semantic retrieval."
)


question = st.text_input(
    "Enter your question:",
    placeholder=(
        "Example: How many Critical tickets are there?"
    )
)


# ==================================================
# 13. PROCESS USER QUESTION
# ==================================================

if st.button(
    "Ask",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ------------------------------------------
        # Determine route
        # ------------------------------------------

        route = determine_route(
            question
        )


        # ==========================================
        # PANDAS ROUTE
        # ==========================================

        if route == "PANDAS":

            st.info(
                "📊 Route selected: "
                "Pandas Data Analysis"
            )

            with st.spinner(
                "Analysing complete dataset..."
            ):

                answer = pandas_answer(
                    question
                )


            st.subheader("💡 Answer")

            st.markdown(answer)


        # ==========================================
        # RAG ROUTE
        # ==========================================

        else:

            st.info(
                "🔎 Route selected: "
                "RAG Semantic Search"
            )

            with st.spinner(
                "Searching ChromaDB..."
            ):

                answer, results = (
                    rag_answer(question)
                )


            st.subheader("💡 Answer")

            st.write(answer)


            # --------------------------------------
            # Retrieved evidence
            # --------------------------------------

            st.subheader(
                "📄 Retrieved Evidence"
            )

            st.caption(
                "These records were retrieved from "
                "ChromaDB and supplied to the LLM."
            )


            for i, result in enumerate(
                results,
                start=1
            ):

                ticket_id = (
                    result.metadata.get(
                        "ticket_id",
                        "Unknown"
                    )
                )

                priority = (
                    result.metadata.get(
                        "priority",
                        "Unknown"
                    )
                )

                topic = (
                    result.metadata.get(
                        "topic",
                        "Unknown"
                    )
                )


                title = (
                    f"Result {i}: "
                    f"{ticket_id} | "
                    f"{priority} | "
                    f"{topic}"
                )


                with st.expander(title):

                    st.write(
                        result.page_content
                    )

                    st.write(
                        "**Metadata**"
                    )

                    st.json(
                        result.metadata
                    )