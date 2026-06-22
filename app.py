import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide"
)

st.markdown("""
<style>
.anime-card {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
    color: white;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #FF4B4B;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #BBBBBB;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    df = pd.read_csv("anime.csv")

    df["genre"] = df["genre"].fillna("Unknown")

    cv = CountVectorizer(stop_words="english")
    genre_matrix = cv.fit_transform(df["genre"])

    return df, genre_matrix


df, genre_matrix = load_data()


def recommend_anime(anime_name):

    if anime_name not in df["name"].values:
        return []

    index = df[df["name"] == anime_name].index[0]

    similarity_scores = cosine_similarity(
        genre_matrix[index],
        genre_matrix
    )[0]

    distances = list(enumerate(similarity_scores))

    sorted_distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in sorted_distances:
        anime_title = df.iloc[i[0]]["name"]

        if anime_title != anime_name:
            recommendations.append(anime_title)

        if len(recommendations) == 10:
            break

    return recommendations


st.markdown(
    '<div class="title">🎌 Anime Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find anime similar to your favorite one</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:
    selected_anime = st.selectbox(
        "Choose an Anime",
        sorted(df["name"].unique())
    )

with col2:
    st.write("")
    st.write("")
    recommend_btn = st.button(
        "🚀 Get Recommendations",
        use_container_width=True
    )

if recommend_btn:

    with st.spinner("Finding recommendations..."):
        recommendations = recommend_anime(selected_anime)

    st.success(f"Top recommendations for: {selected_anime}")

    cols = st.columns(2)

    for idx, anime in enumerate(recommendations):

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="anime-card">
                    ⭐ {anime}
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("""
<hr>
<div style='text-align:center;color:gray'>
Made with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
