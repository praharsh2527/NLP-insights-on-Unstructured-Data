import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Amazon Food Reviews — NLP Dashboard",
    page_icon="🍽️",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('../datasets/Reviews_topics.csv')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Year'] = df['Time'].dt.year
    return df

df = load_data()

@st.cache_data
def load_raw_data():
    return pd.read_csv('../datasets/Reviews_cleaned.csv')

df_raw = load_raw_data()

st.sidebar.title("🍽️ Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "💬 Sentiment Analysis",
    "🗂️ Topic Explorer",
    "📈 Trends"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Reviews:** {len(df):,}")
st.sidebar.markdown(f"**Years Covered:** 1999–2012")
st.sidebar.markdown(f"**Products:** 74,258")

# ─── PAGE 1: OVERVIEW ───────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Amazon Fine Food Reviews — Overview")
    st.markdown("Exploratory analysis of **364,163 reviews** across **74,258 products** from 1999–2012.")

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", "364,163")
    col2.metric("Unique Products", "74,258")
    col3.metric("Unique Users", "256,059")
    col4.metric("Avg Sentiment", "84% Positive")

    st.markdown("---")

    # Review Volume Over Time
    st.subheader("📈 Review Volume Over Time")
    reviews_per_year = df.groupby('Year').size().reset_index(name='count')
    fig1 = px.area(reviews_per_year, x='Year', y='count',
                   title='Number of Reviews Per Year',
                   color_discrete_sequence=['steelblue'])
    st.plotly_chart(fig1, width='stretch')

    st.markdown("---")

    # Score Distribution
    col1, col2 = st.columns(2)

    with col1:
      st.subheader("⭐ Score Distribution")
     # Load original data for score distribution (has all 5 stars)
      df_raw = pd.read_csv('../datasets/Reviews_cleaned.csv')
      score_counts = df_raw['Score'].value_counts().sort_index()
      fig2 = px.bar(x=score_counts.index, y=score_counts.values,
                  labels={'x': 'Score', 'y': 'Count'},
                  color_discrete_sequence=['steelblue'])
      st.plotly_chart(fig2, width='stretch')

    with col2:
        st.subheader("📝 Review Length Distribution")
        fig3 = px.histogram(df, x='review_length', nbins=50,
                            range_x=[0, 300],
                            color_discrete_sequence=['mediumseagreen'])
        st.plotly_chart(fig3, width='stretch')

# ─── PAGE 2: SENTIMENT ANALYSIS ─────────────────────────────────
elif page == "💬 Sentiment Analysis":
    st.title("💬 Sentiment Analysis")
    st.markdown("Binary sentiment classification using **TF-IDF + Logistic Regression** vs **VADER**.")

    st.markdown("---")

    # Model Performance Metrics
    st.subheader("🏆 Model Performance Comparison")
    col1, col2, col3 = st.columns(3)
    col1.metric("LR Accuracy", "90.93%", "+3.29% vs VADER")
    col2.metric("LR Negative Recall", "88%", "+45% vs VADER Raw")
    col3.metric("LR Positive Precision", "98%")

    st.markdown("---")

    # Comparison Chart
    st.subheader("📊 LR vs VADER Comparison")
    models = ['Logistic Regression', 'VADER (Cleaned)', 'VADER (Raw)']
    accuracy = [0.9093, 0.8664, 0.8759]
    neg_recall = [0.88, 0.33, 0.43]
    pos_recall = [0.91, 0.96, 0.96]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Overall Accuracy', x=models, y=accuracy,
                         marker_color='steelblue'))
    fig.add_trace(go.Bar(name='Negative Recall', x=models, y=neg_recall,
                         marker_color='tomato'))
    fig.add_trace(go.Bar(name='Positive Recall', x=models, y=pos_recall,
                         marker_color='mediumseagreen'))
    fig.update_layout(barmode='group', yaxis_range=[0, 1.1],
                      title='Model Performance Comparison')
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Sentiment Distribution
    st.subheader("📈 Sentiment Distribution")
    sent_counts = df['sentiment'].value_counts()
    fig2 = px.pie(values=sent_counts.values,
                  names=['Positive', 'Negative'],
                  color_discrete_sequence=['mediumseagreen', 'tomato'],
                  title='Overall Sentiment Split')
    st.plotly_chart(fig2, width='stretch')

    st.markdown("---")

    # Key Insights
    st.subheader("💡 Key Insights")
    st.info("🔵 Logistic Regression outperforms VADER on every metric — especially Negative Recall (88% vs 43%)")
    st.warning("🟠 VADER struggles to detect negative reviews when stopwords are removed — always use raw text with VADER")
    st.success("🟢 Both models agree: 84% of Amazon food reviews are positive")


# ─── PAGE 3: TOPIC EXPLORER ──────────────────────────────────────
elif page == "🗂️ Topic Explorer":
    st.title("🗂️ Topic Explorer")
    st.markdown("**LDA Topic Modeling** on 364,163 reviews — 8 automatically discovered topics.")

    st.markdown("---")

    # Topic Distribution
    st.subheader("📊 Reviews Per Topic")
    topic_counts = df['topic_label'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig1 = px.bar(topic_counts, x='Count', y='Topic',
                  orientation='h',
                  color='Count',
                  color_continuous_scale='viridis',
                  title='Number of Reviews per Topic')
    st.plotly_chart(fig1, width='stretch')

    st.markdown("---")

    # Topic vs Sentiment
    st.subheader("💬 Sentiment Rate by Topic")
    topic_sentiment = df.groupby('topic_label')['sentiment'].agg(
        total='count', positive_count='sum'
    )
    topic_sentiment['positive_rate'] = (
        topic_sentiment['positive_count'] / topic_sentiment['total'] * 100
    ).round(2)
    topic_sentiment['negative_rate'] = (
        100 - topic_sentiment['positive_rate']
    ).round(2)
    topic_sentiment = topic_sentiment.sort_values('positive_rate').reset_index()

    fig2 = px.bar(topic_sentiment, x='positive_rate', y='topic_label',
                  orientation='h',
                  color='positive_rate',
                  color_continuous_scale='RdYlGn',
                  title='Positive Sentiment Rate by Topic (%)')
    fig2.add_vline(x=80, line_dash='dash', line_color='red',
                   annotation_text='80% threshold')
    st.plotly_chart(fig2, width='stretch')

    st.markdown("---")

    # Key Insights
    st.subheader("💡 Key Insights")
    st.error("🚨 Packaging & Product Issues — 40.88% negative rate. Highest complaint category.")
    st.warning("⚠️ Pantry & Cooking Ingredients — 22.45% negative rate. Declining quality signals.")
    st.success("✅ Snacks & Sweet Foods — 91.85% positive. Amazon's strongest food category.")


# ─── PAGE 4: TRENDS ──────────────────────────────────────────────
elif page == "📈 Trends":
    st.title("📈 Trends Over Time")
    st.markdown("How review volume and sentiment evolved from **1999 to 2012**.")

    st.markdown("---")

    # Sentiment Over Time
    st.subheader("💬 Positive Sentiment Rate Over Time")
    sentiment_by_year = df.groupby('Year')['sentiment'].agg(
        positive_rate=lambda x: (x == 1).mean() * 100
    ).reset_index()
    fig1 = px.line(sentiment_by_year, x='Year', y='positive_rate',
                   markers=True,
                   color_discrete_sequence=['mediumseagreen'],
                   title='Positive Sentiment Rate 1999–2012')
    fig1.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig1, width='stretch')

    st.markdown("---")

    # Topic Over Time
    st.subheader("🗂️ Topic Popularity Over Time")
    topic_year = df[df['Year'] >= 2006].groupby(
        ['Year', 'topic_label']
    ).size().reset_index(name='count')

    fig2 = px.line(topic_year, x='Year', y='count',
                   color='topic_label',
                   markers=True,
                   title='Topic Volume Over Time (2006–2012)')
    st.plotly_chart(fig2, width='stretch')

    st.markdown("---")

    # Key Insights
    st.subheader("💡 Key Insights")
    st.warning("📉 Positive sentiment declined from 93% (2003) to 82% (2012) as platform scaled")
    st.info("🛒 Shopping & Pricing broke away from all other topics in 2010 — price sensitivity grew")
    st.error("📦 Packaging complaints grew consistently — a structural unresolved issue")