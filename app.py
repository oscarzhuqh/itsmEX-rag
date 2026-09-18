import os
import streamlit as st
import pandas as pd
import altair as alt

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

CSV_PATH = "data/ITSM_Dataset.csv"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "itsm_tickets"

# Number of tickets used for deployed RAG demonstration
RAG_INDEX_SIZE = 1000

# Number of documents inserted into Chroma per batch
BATCH_SIZE = 100


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
    "Explore ITSM statistics and ask questions using "
    "traditional data analysis or Retrieval-Augmented "
    "Generation (RAG)."
)


# ==================================================
# 3. LOAD COMPLETE DATASET
# ==================================================

@st.cache_data
def load_data():

    return pd.read_csv(CSV_PATH)


df = load_data()


# ==================================================
# 4. OPENAI EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


embedding_model = load_embedding_model()


# ==================================================
# 5. CREATE LANGCHAIN DOCUMENTS
# ==================================================

def create_documents(dataframe):

    documents = []

    for _, row in dataframe.iterrows():

        text = f"""
Ticket ID: {row['Ticket ID']}
Status: {row['Status']}
Priority: {row['Priority']}
Source: {row['Source']}
Topic: {row['Topic']}
Agent Group: {row['Agent Group']}
Agent Name: {row['Agent Name']}
"""

        document = Document(
            page_content=text,
            metadata={
                "ticket_id": str(row["Ticket ID"]),
                "priority": str(row["Priority"]),
                "status": str(row["Status"]),
                "topic": str(row["Topic"])
            }
        )

        documents.append(document)

    return documents


# ==================================================
# 6. LOAD OR CREATE CHROMADB
# ==================================================

@st.cache_resource
def load_or_create_vectorstore():

    # ------------------------------------------------
    # Create connection to ChromaDB
    # ------------------------------------------------

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH
    )

    # ------------------------------------------------
    # Check how many documents already exist
    # ------------------------------------------------

    existing_count = vectorstore._collection.count()

    # ------------------------------------------------
    # If database already contains documents,
    # simply use it
    # ------------------------------------------------

    if existing_count > 0:

        return vectorstore, existing_count


    # ------------------------------------------------
    # Otherwise create demonstration RAG index
    # ------------------------------------------------

    rag_df = df.head(RAG_INDEX_SIZE)

    documents = create_documents(rag_df)

    progress_text = st.empty()

    progress_bar = st.progress(0)

    progress_text.info(
        "Creating RAG demonstration index. "
        "This is required only when the vector "
        "database is not available."
    )


    total_documents = len(documents)


    for start in range(
        0,
        total_documents,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_documents
        )

        batch = documents[start:end]

        vectorstore.add_documents(batch)

        progress = end / total_documents

        progress_bar.progress(progress)

        progress_text.info(
            f"Creating RAG index: "
            f"{end:,} / {total_documents:,} tickets"
        )


    progress_bar.empty()

    progress_text.empty()


    return vectorstore, total_documents


# ==================================================
# 7. INITIALISE VECTOR DATABASE
# ==================================================

try:

    vectorstore, rag_ticket_count = (
        load_or_create_vectorstore()
    )

    rag_available = True

except Exception as error:

    rag_available = False

    rag_ticket_count = 0

    st.warning(
        "The RAG vector database could not be "
        "initialised. Data analytics remains available."
    )

    st.caption(
        f"RAG error: {error}"
    )


# ==================================================
# 8. LOAD LLM
# ==================================================

@st.cache_resource
def load_llm():

    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )


llm = load_llm()


# ==================================================
# 9. SYSTEM COVERAGE
# ==================================================

st.subheader("📁 System Coverage")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Dataset",
        f"{len(df):,}",
        help="Total number of tickets in the CSV dataset."
    )


with col2:

    st.metric(
        "Analytics Coverage",
        f"{len(df):,}",
        help=(
            "Pandas performs statistical analysis "
            "against the complete CSV dataset."
        )
    )


with col3:

    if rag_available:

        st.metric(
            "RAG Index",
            f"{rag_ticket_count:,}",
            help=(
                "Number of tickets currently indexed "
                "in ChromaDB for semantic retrieval."
            )
        )

    else:

        st.metric(
            "RAG Index",
            "Unavailable"
        )


st.caption(
    "📊 Pandas analytics uses the complete dataset. "
    "🔎 RAG semantic search uses the tickets indexed "
    "in ChromaDB."
)


# ==================================================
# 10. PRIORITY ANALYTICS DASHBOARD
# ==================================================

st.divider()

st.subheader("📊 ITSM Priority Analytics")

st.write(
    "Explore the distribution of priority levels "
    "across the complete ITSM dataset."
)


show_dashboard = st.toggle(
    "Show Priority Dashboard"
)


if show_dashboard:

    # ------------------------------------------------
    # Priority statistics
    # ------------------------------------------------

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


    # ------------------------------------------------
    # KPI cards
    # ------------------------------------------------

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


    # ------------------------------------------------
    # Coloured priority chart
    # ------------------------------------------------

    st.subheader("Priority Distribution")


    priority_chart = pd.DataFrame({
        "Priority": priority_counts.index,
        "Number of Tickets": priority_counts.values
    })


    chart = (
        alt.Chart(priority_chart)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5
        )
        .encode(

            x=alt.X(
                "Priority:N",
                sort=[
                    "Critical",
                    "High",
                    "Medium",
                    "Low"
                ],
                title="Priority"
            ),

            y=alt.Y(
                "Number of Tickets:Q",
                title="Number of Tickets"
            ),

            color=alt.Color(
                "Priority:N",

                scale=alt.Scale(
                    domain=[
                        "Critical",
                        "High",
                        "Medium",
                        "Low"
                    ],

                    range=[
                        "#E63946",
                        "#F77F00",
                        "#F4C430",
                        "#2ECC71"
                    ]
                ),

                legend=alt.Legend(
                    title="Priority"
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "Priority:N",
                    title="Priority"
                ),

                alt.Tooltip(
                    "Number of Tickets:Q",
                    title="Tickets",
                    format=","
                )
            ]
        )
        .properties(
            height=400
        )
    )


    st.altair_chart(
        chart,
        width="stretch"
    )


    # ------------------------------------------------
    # Priority statistics table
    # ------------------------------------------------

    st.subheader("Priority Statistics")


    priority_statistics = pd.DataFrame({

        "Priority":
            priority_counts.index,

        "Tickets":
            priority_counts.values,

        "Percentage (%)":
            priority_percentages.values
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


    # ------------------------------------------------
    # Critical + High workload
    # ------------------------------------------------

    urgent_tickets = (
        priority_counts["Critical"]
        + priority_counts["High"]
    )


    urgent_percentage = (
        urgent_tickets
        / total_tickets
        * 100
    )


    st.subheader(
        "⚠️ High-Priority Workload"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Critical + High Tickets",
            f"{urgent_tickets:,}"
        )


    with col2:

        st.metric(
            "Percentage of All Tickets",
            f"{urgent_percentage:.2f}%"
        )


# ==================================================
# 11. QUERY ROUTER
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
# 12. PANDAS ANALYTICAL ENGINE
# ==================================================

def pandas_answer(question):

    question_lower = question.lower()


    # ------------------------------------------------
    # Priority count
    # ------------------------------------------------

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


    # ------------------------------------------------
    # Status count
    # ------------------------------------------------

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


    # ------------------------------------------------
    # Total tickets
    # ------------------------------------------------

    if (
        "total" in question_lower
        or
        "how many tickets" in question_lower
    ):

        return (
            f"The complete dataset contains "
            f"**{len(df):,} tickets**."
        )


    # ------------------------------------------------
    # Unsupported analytics
    # ------------------------------------------------

    return (
        "This appears to be an analytical question, "
        "but the current Pandas engine does not yet "
        "support this calculation."
    )


# ==================================================
# 13. RAG ENGINE
# ==================================================

def rag_answer(question):

    # ------------------------------------------------
    # Retrieve five semantically similar tickets
    # ------------------------------------------------

    results = (
        vectorstore.similarity_search(
            question,
            k=5
        )
    )


    if not results:

        return (
            "No relevant tickets were retrieved.",
            []
        )


    # ------------------------------------------------
    # Combine retrieved records
    # ------------------------------------------------

    context = "\n\n".join(

        result.page_content

        for result in results

    )


    # ------------------------------------------------
    # RAG prompt
    # ------------------------------------------------

    prompt = f"""
You are an ITSM support assistant.

Answer the user's question using ONLY the ITSM
ticket records contained in the CONTEXT below.

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
   the entire ITSM dataset.

7. Remember that RAG retrieves only the most
   semantically relevant records from the indexed
   ticket collection.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


    response = llm.invoke(
        prompt
    )


    return (
        response.content,
        results
    )


# ==================================================
# 14. ITSM ASSISTANT
# ==================================================

st.divider()

st.subheader(
    "🤖 Ask the ITSM Assistant"
)


st.write(
    "The Query Router automatically selects either "
    "Pandas data analysis or RAG semantic search."
)


question = st.text_input(

    "Enter your question:",

    placeholder=(
        "Example: Find tickets related "
        "to network connectivity problems"
    )
)


# ==================================================
# 15. PROCESS QUESTION
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

        route = determine_route(
            question
        )


        # ==========================================
        # PANDAS
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


            st.subheader(
                "💡 Answer"
            )


            st.markdown(
                answer
            )


            st.caption(
                f"Analysis coverage: "
                f"{len(df):,} CSV records."
            )


        # ==========================================
        # RAG
        # ==========================================

        else:

            if not rag_available:

                st.error(
                    "RAG is currently unavailable. "
                    "Please check the OpenAI API key "
                    "and vector database configuration."
                )


            else:

                st.info(
                    "🔎 Route selected: "
                    "RAG Semantic Search"
                )


                with st.spinner(
                    "Searching ChromaDB..."
                ):

                    answer, results = (
                        rag_answer(
                            question
                        )
                    )


                st.subheader(
                    "💡 Answer"
                )


                st.write(
                    answer
                )


                st.caption(
                    f"RAG search coverage: "
                    f"{rag_ticket_count:,} "
                    f"indexed tickets."
                )


                # ----------------------------------
                # Retrieved evidence
                # ----------------------------------

                st.subheader(
                    "📄 Retrieved Evidence"
                )


                st.caption(
                    "These records were retrieved "
                    "from ChromaDB and supplied "
                    "to the LLM."
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


                    with st.expander(
                        title
                    ):

                        st.write(
                            result.page_content
                        )


                        st.write(
                            "**Metadata**"
                        )


                        st.json(
                            result.metadata
                        )
