"""
Streamlit app for Cross-Linguistic Morphology Explorer.
Converts R Shiny app.R to Python Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import gdown
import tempfile


# =======================
# Configuration
# =======================
st.set_page_config(
    page_title="Cross-Linguistic Morphology Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Load Data
# =======================
@st.cache_data
def load_data():
    """Load the stem-labeled dataset."""
    
    # Temporary directory in Streamlit Cloud
    temp_dir = tempfile.mkdtemp()

    # Path for the file in the temporary directory
    pkl_path = os.path.join(temp_dir, 'df_stem_labeled.pkl')
    
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    else:
        # Fallback: try to download from Google Drive if not found locally
        st.info("Data file not found locally. Downloading from Google Drive...")

        # URL of your file in Google Drive (replace with your actual file ID)
        file_id = 'your-google-file-id-here'  # Replace with the actual file ID
        url = f'https://drive.google.com/uc?id={file_id}'

        try:
            # Download the file using gdown to the temporary directory
            gdown.download(url, pkl_path, quiet=False)
            st.success("File downloaded successfully!")
            
            # Load the data after download
            df = pd.read_pickle(pkl_path)
        except Exception as e:
            st.error(f"Error downloading the file: {e}")
            st.stop()
    
    # Select relevant columns
    df = df[['Language_ID', 'wd', 'mb', 'morph_label', 'gloss', 'coarse_pos']].copy()

    return df


df = load_data()

# =======================
# Title
# =======================
st.title("Cross-Linguistic Morphology Explorer")

# =======================
# Sidebar Filters
# =======================
st.sidebar.header("Filters")

# Analysis type
analysis_type = st.sidebar.selectbox(
    "Analysis Type:",
    options=[
        "gloss_lang",
        "gloss_pos",
        "length_pos",
        "morph_pos"
    ],
    format_func=lambda x: {
        "gloss_lang": "Gloss Distribution Across Languages",
        "gloss_pos": "POS Distribution for Selected Gloss",
        "length_pos": "Word Length vs POS",
        "morph_pos": "Morphological Label × POS"
    }.get(x, x),
    key="analysis_type"
)

st.sidebar.divider()

# Language filter
all_languages = sorted(df['Language_ID'].unique())
filter_language = st.sidebar.multiselect(
    "Language:",
    options=all_languages,
    default=all_languages,
    key="filter_language"
)

st.sidebar.divider()

# Gloss keyword + picker (dynamic based on language filter)
gloss_keyword = st.sidebar.text_input(
    "Search Gloss (keyword / regex):",
    placeholder="e.g., run | eat | water",
    key="gloss_keyword"
)

# Get available glosses for selected languages
df_lang_filtered = df[df['Language_ID'].isin(filter_language)]

if gloss_keyword:
    # Filter glosses by keyword
    gloss_options = sorted(
        df_lang_filtered[
            df_lang_filtered['gloss'].str.contains(gloss_keyword, case=False, na=False)
        ]['gloss'].dropna().unique()
    )
    # Auto-select all available glosses when keyword search is performed
    default_gloss = gloss_options
else:
    gloss_options = sorted(df_lang_filtered['gloss'].dropna().unique())
    default_gloss = []

filter_gloss = st.sidebar.multiselect(
    "Gloss:",
    options=gloss_options,
    default=default_gloss,
    key="filter_gloss"
)

st.sidebar.divider()

# POS filter (only show if gloss is selected)
if len(filter_gloss) > 0:
    df_gloss_filtered = df_lang_filtered[df_lang_filtered['gloss'].isin(filter_gloss)]
    pos_options = sorted(df_gloss_filtered['coarse_pos'].dropna().unique())
    
    filter_pos = st.sidebar.multiselect(
        "POS category (coarse_pos):",
        options=pos_options,
        default=pos_options,
        key="filter_pos"
    )
else:
    filter_pos = []

# Morph filter (only show if POS is selected)
if len(filter_gloss) > 0 and len(filter_pos) > 0:
    df_pos_filtered = df_gloss_filtered[df_gloss_filtered['coarse_pos'].isin(filter_pos)]
    morph_options = sorted(df_pos_filtered['morph_label'].dropna().unique())
    
    filter_morph = st.sidebar.multiselect(
        "Morph Label:",
        options=morph_options,
        default=morph_options,
        key="filter_morph"
    )
else:
    filter_morph = []

# =======================
# Data Filtering
# =======================
def get_filtered_data():
    """Apply all filters to the dataframe."""
    if len(filter_gloss) == 0:
        return pd.DataFrame()
    
    data = df[
        (df['Language_ID'].isin(filter_language)) &
        (df['gloss'].isin(filter_gloss))
    ].copy()
    
    if len(filter_pos) > 0:
        data = data[data['coarse_pos'].isin(filter_pos)]
    
    if len(filter_morph) > 0:
        data = data[data['morph_label'].isin(filter_morph)]
    
    return data


filtered_data = get_filtered_data()

# =======================
# Title for Analysis
# =======================
analysis_titles = {
    "gloss_lang": "Gloss Distribution Across Languages",
    "gloss_pos": "POS Distribution for Selected Gloss",
    "length_pos": "Word Length vs POS",
    "morph_pos": "Morphological Label × POS"
}

st.header(analysis_titles.get(analysis_type, "Analysis"))

# =======================
# Main Content
# =======================
if len(filter_gloss) == 0:
    st.warning("Please select at least one gloss to proceed.")
    st.stop()

if len(filtered_data) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# Display summary stats in columns
col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
with col_stats1:
    st.metric("Total Records", len(filtered_data))
with col_stats2:
    st.metric("Languages", filtered_data['Language_ID'].nunique())
with col_stats3:
    st.metric("Glosses", filtered_data['gloss'].nunique())
with col_stats4:
    st.metric("POS Tags", filtered_data['coarse_pos'].nunique())

st.divider()

# Show option for gloss_lang analysis - place in columns with plot options
col_options_left, col_options_right = st.columns([1, 1])

with col_options_left:
    if analysis_type == "gloss_lang":
        split_gloss = st.checkbox("Breakdown by Gloss", value=True)
    else:
        split_gloss = False

with col_options_right:
    if analysis_type == "gloss_pos":
        split_lang = st.checkbox("Breakdown by Language", value=False)
    else:
        split_lang = False

# =======================
# Plot
# =======================
if analysis_type == "gloss_lang":
    if split_gloss:
        # Breakdown by gloss
        plot_data = filtered_data.groupby(['Language_ID', 'gloss']).size().reset_index(name='count')
        fig = px.bar(
            plot_data,
            x='Language_ID',
            y='count',
            color='gloss',
            barmode='group',
            labels={'Language_ID': 'Language', 'count': 'Count'},
            title="Gloss Distribution Across Languages (by Gloss)"
        )
    else:
        # Total count per language
        plot_data = filtered_data.groupby('Language_ID').size().reset_index(name='count')
        fig = px.bar(
            plot_data,
            x='Language_ID',
            y='count',
            labels={'Language_ID': 'Language', 'count': 'Total Count (all selected gloss)'},
            color_discrete_sequence=['#2C3E50']
        )
    
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "gloss_pos":
    if split_lang:
        # Breakdown by language
        plot_data = filtered_data.groupby(['coarse_pos', 'Language_ID']).size().reset_index(name='count')
        fig = px.bar(
            plot_data,
            x='coarse_pos',
            y='count',
            color='Language_ID',
            barmode='group',
            labels={'coarse_pos': 'POS', 'count': 'Count', 'Language_ID': 'Language'},
            title="POS Distribution (by Language)"
        )
    else:
        # Stacked by POS across all languages
        plot_data = filtered_data.groupby(['Language_ID', 'coarse_pos']).size().reset_index(name='count')
        fig = px.bar(
            plot_data,
            x='Language_ID',
            y='count',
            color='coarse_pos',
            barmode='stack',
            labels={'Language_ID': 'Language', 'count': 'Count', 'coarse_pos': 'POS'},
            title="POS Distribution for Selected Gloss"
        )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "length_pos":
    filtered_data_copy = filtered_data.copy()
    filtered_data_copy['word_length'] = filtered_data_copy['wd'].str.len()
    
    fig = px.box(
        filtered_data_copy,
        x='coarse_pos',
        y='word_length',
        color='coarse_pos',
        labels={'coarse_pos': 'POS', 'word_length': 'Word Length'},
        title="Word Length Distribution by POS"
    )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "morph_pos":
    plot_data = filtered_data.groupby(['coarse_pos', 'morph_label']).size().reset_index(name='count')
    fig = px.bar(
        plot_data,
        x='coarse_pos',
        y='count',
        color='morph_label',
        barmode='stack',
        labels={'coarse_pos': 'POS', 'count': 'Count', 'morph_label': 'Morph Label'},
        title="Morphological Label × POS Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# =======================
# Table
# =======================
st.divider()

# Use expander to collapse the table by default
with st.expander("📊 Data Table (click to expand)", expanded=False):
    if analysis_type == "gloss_lang":
        if split_gloss:
            table_data = filtered_data.groupby(['Language_ID', 'gloss']).size().reset_index(name='count')
        else:
            table_data = filtered_data.groupby('Language_ID').size().reset_index(name='count')

    elif analysis_type == "gloss_pos":
        if split_lang:
            table_data = filtered_data.groupby(['coarse_pos', 'Language_ID']).size().reset_index(name='count')
        else:
            table_data = filtered_data.groupby(['Language_ID', 'coarse_pos']).size().reset_index(name='count')

    elif analysis_type == "length_pos":
        table_data = filtered_data.copy()
        table_data['word_length'] = table_data['wd'].str.len()
        table_data = table_data[['wd', 'coarse_pos', 'word_length']]

    elif analysis_type == "morph_pos":
        table_data = filtered_data.groupby(['coarse_pos', 'morph_label']).size().reset_index(name='count')

    st.dataframe(table_data, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(table_data)} rows from {len(filtered_data)} total records")
