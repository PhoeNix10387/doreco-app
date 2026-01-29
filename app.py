import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os
import gdown

st.set_page_config(
    page_title="Cross-Linguistic Morphology Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    temp_dir = tempfile.mkdtemp()
    pkl_path = os.path.join(temp_dir, "df_stem_labeled.pkl")

    if not os.path.exists(pkl_path):
        file_id = "1yP6iTubQZ_wtlbJxqngWCwYz4pv6NsfO"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, pkl_path, quiet=True)

    df = pd.read_pickle(pkl_path)
    return df[["Language_ID", "wd", "mb", "morph_label", "gloss", "coarse_pos"]].copy()

df = load_data()

st.title("Cross-Linguistic Morphology Explorer")

st.sidebar.header("Filters")

analysis_type = st.sidebar.selectbox(
    "Analysis Type:",
    ["gloss_lang", "gloss_pos", "length_pos", "morph_pos"],
    format_func=lambda x: {
        "gloss_lang": "Gloss Distribution Across Languages",
        "gloss_pos": "POS Distribution for Selected Gloss",
        "length_pos": "Word Length vs POS",
        "morph_pos": "Morphological Label × POS"
    }[x]
)

st.sidebar.divider()

languages = sorted(df["Language_ID"].unique())
filter_language = st.sidebar.multiselect(
    "Language:",
    languages,
    default=languages
)

st.sidebar.divider()

gloss_keyword = st.sidebar.text_input(
    "Search Gloss (keyword / regex):",
    placeholder="e.g., run | eat | water"
)

df_lang = df[df["Language_ID"].isin(filter_language)]

if gloss_keyword:
    gloss_options = sorted(
        df_lang[
            df_lang["gloss"].str.contains(gloss_keyword, case=False, na=False)
        ]["gloss"].dropna().unique()
    )

    if st.session_state.get("_last_gloss_keyword") != gloss_keyword:
        st.session_state.filter_gloss = gloss_options
        st.session_state._last_gloss_keyword = gloss_keyword
else:
    gloss_options = sorted(df_lang["gloss"].dropna().unique())

filter_gloss = st.sidebar.multiselect(
    "Gloss:",
    gloss_options,
    key="filter_gloss"
)

st.sidebar.divider()

if filter_gloss:
    df_gloss = df_lang[df_lang["gloss"].isin(filter_gloss)]
    pos_options = sorted(df_gloss["coarse_pos"].dropna().unique())
    filter_pos = st.sidebar.multiselect(
        "POS category:",
        pos_options,
        default=pos_options
    )
else:
    filter_pos = []

if filter_gloss and filter_pos:
    df_pos = df_gloss[df_gloss["coarse_pos"].isin(filter_pos)]
    morph_options = sorted(df_pos["morph_label"].dropna().unique())
    filter_morph = st.sidebar.multiselect(
        "Morph Label:",
        morph_options,
        default=morph_options
    )
else:
    filter_morph = []

def get_filtered_data():
    if not filter_gloss:
        return pd.DataFrame()

    data = df[
        df["Language_ID"].isin(filter_language) &
        df["gloss"].isin(filter_gloss)
    ]

    if filter_pos:
        data = data[data["coarse_pos"].isin(filter_pos)]
    if filter_morph:
        data = data[data["morph_label"].isin(filter_morph)]

    return data

filtered_data = get_filtered_data()

titles = {
    "gloss_lang": "Gloss Distribution Across Languages",
    "gloss_pos": "POS Distribution for Selected Gloss",
    "length_pos": "Word Length vs POS",
    "morph_pos": "Morphological Label × POS"
}

st.header(titles[analysis_type])

if filtered_data.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", len(filtered_data))
c2.metric("Languages", filtered_data["Language_ID"].nunique())
c3.metric("Glosses", filtered_data["gloss"].nunique())
c4.metric("POS Tags", filtered_data["coarse_pos"].nunique())

st.divider()

if analysis_type == "gloss_lang":
    split_gloss = st.checkbox("Breakdown by Gloss", value=True)

    if split_gloss:
        plot_data = filtered_data.groupby(
            ["Language_ID", "gloss"]
        ).size().reset_index(name="count")

        fig = px.bar(
            plot_data,
            x="Language_ID",
            y="count",
            color="gloss",
            barmode="group"
        )
    else:
        plot_data = filtered_data.groupby(
            "Language_ID"
        ).size().reset_index(name="count")

        fig = px.bar(
            plot_data,
            x="Language_ID",
            y="count"
        )

elif analysis_type == "gloss_pos":
    split_lang = st.checkbox("Breakdown by Language", value=False)

    if split_lang:
        plot_data = filtered_data.groupby(
            ["coarse_pos", "Language_ID"]
        ).size().reset_index(name="count")

        fig = px.bar(
            plot_data,
            x="coarse_pos",
            y="count",
            color="Language_ID",
            barmode="group"
        )
    else:
        plot_data = filtered_data.groupby(
            ["Language_ID", "coarse_pos"]
        ).size().reset_index(name="count")

        fig = px.bar(
            plot_data,
            x="Language_ID",
            y="count",
            color="coarse_pos",
            barmode="stack"
        )

elif analysis_type == "length_pos":
    tmp = filtered_data.copy()
    tmp["word_length"] = tmp["wd"].str.len()

    fig = px.box(
        tmp,
        x="coarse_pos",
        y="word_length",
        color="coarse_pos"
    )

elif analysis_type == "morph_pos":
    plot_data = filtered_data.groupby(
        ["coarse_pos", "morph_label"]
    ).size().reset_index(name="count")

    fig = px.bar(
        plot_data,
        x="coarse_pos",
        y="count",
        color="morph_label",
        barmode="stack"
    )

st.plotly_chart(fig, use_container_width=True)

st.divider()

with st.expander("📊 Data Table", expanded=False):
    st.dataframe(filtered_data, use_container_width=True, hide_index=True)
