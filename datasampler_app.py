import streamlit as st
import pandas as pd
import numpy as np
import os
import io

# Constants
MAX_FILE_SIZE_MB = 200
MAX_SAMPLE_ROWS = 75000
MIN_SAMPLE_ROWS = 500
MAX_SAMPLE_PCT = 0.75

# App title and description
st.set_page_config(page_title="DataSampler", layout="wide")
st.title("📦 DataSampler")
st.markdown("Use DataSampler to create smaller, analysis-ready datasets from large files.")
st.info("🔍 **Need to analyze your sampled data?** Use our companion tool [DataWizard](https://datawizardtool.streamlit.app/)!")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'sampled_df' not in st.session_state:
    st.session_state.sampled_df = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = {}
if 'sample_info' not in st.session_state:
    st.session_state.sample_info = {}
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Reset function
def reset_app():
    st.session_state.clear()
    st.rerun()

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# Automatically reset everything if file is removed
if uploaded_file is None and st.session_state.get("df") is not None:
    reset_app()

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large! Limit is {MAX_FILE_SIZE_MB} MB.")
    else:
        try:
            if uploaded_file.name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            else:
                df = pd.read_excel(uploaded_file)

            rows, cols = df.shape
            st.session_state.df = df
            st.session_state.file_info = {
                "name": uploaded_file.name,
                "size_mb": round(file_size_mb, 2),
                "rows": rows,
                "cols": cols,
                "missing": df.isnull().sum().sum()
            }
            st.session_state.uploaded_file = uploaded_file

            if rows < MIN_SAMPLE_ROWS:
                st.warning(f"File contains only {rows} rows. Minimum for sampling is {MIN_SAMPLE_ROWS}.")

            st.success("File uploaded successfully!")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
else:
    st.session_state.uploaded_file = None

if st.session_state.get("df") is not None:
    df = st.session_state.get("df")
    row_count = df.shape[0]

    sample_type = st.selectbox("Select sampling method", ["Random", "Stratified", "Systematic"])

    method = sample_type.lower()
    strat_col = None

    cols = st.columns([1, 2])
    with cols[0]:
        sample_size_input = st.radio("Define by", ["Number of rows", "Percentage"], label_visibility="visible")
    with cols[1]:
        if sample_size_input == "Number of rows":
            sample_size = st.number_input("Sample size", min_value=MIN_SAMPLE_ROWS, max_value=MAX_SAMPLE_ROWS, step=500)
        else:
            pct = st.slider("Sample %", min_value=1, max_value=int(MAX_SAMPLE_PCT * 100))
            sample_size = int((pct / 100) * row_count)

    if method == "stratified":
        strat_col = st.selectbox("Stratify by column", df.select_dtypes(include='object').columns.tolist())

    if st.button("🚀 SampleIt"):
        if sample_size > row_count:
            st.error("Sample size cannot exceed total number of rows.")
            st.stop()
        elif sample_size < MIN_SAMPLE_ROWS or sample_size > MAX_SAMPLE_ROWS:
            st.error(f"Sample size must be between {MIN_SAMPLE_ROWS} and {MAX_SAMPLE_ROWS} rows.")
            st.stop()
        elif sample_size > row_count * MAX_SAMPLE_PCT:
            st.error(f"Sample cannot exceed {int(MAX_SAMPLE_PCT * 100)}% of the dataset.")
            st.stop()
        else:
            try:
                if method == "random":
                    sampled_df = df.sample(n=sample_size, random_state=42)
                elif method == "stratified":
                    sampled_df = df.groupby(strat_col, group_keys=False).apply(
                        lambda x: x.sample(frac=min(1, sample_size / row_count), random_state=42)
                    ).head(sample_size)
                elif method == "systematic":
                    step = max(1, row_count // sample_size)
                    sampled_df = df.iloc[::step].head(sample_size)

                # Add DS_SAMPLE column
                tag = f"Sampled[{sample_type}][{len(sampled_df)}/{row_count}]"
                sampled_df["DS_SAMPLE"] = tag

                # Update session
                st.session_state.sampled_df = sampled_df
                st.session_state.sample_info = {
                    "original_rows": row_count,
                    "sample_rows": len(sampled_df),
                    "sample_pct": round(len(sampled_df) / row_count * 100, 2),
                    "sample_type": sample_type
                }

                st.success("Sampling completed successfully!")
                st.dataframe(sampled_df.head(20), use_container_width=True)

                # Download
                csv_buffer = io.StringIO()
                sampled_df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")
                st.download_button(
                    label="⬇️ Download Sampled File",
                    data=csv_bytes,
                    file_name=f"sampled_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Sampling failed: {e}")

# Sidebar (render at end after session state is set)
with st.sidebar:
    st.markdown("## 📊 SampleStats")

    if st.session_state.get("df") is not None and st.session_state.get("file_info"):
        file_info = st.session_state.get("file_info")
        st.markdown("### 📁 File Summary")
        st.write(f"**Name:** {file_info.get('name')}")
        st.write(f"**Size:** {file_info.get('size_mb')} MB")
        st.write(f"**Rows:** {file_info.get('rows')}")
        st.write(f"**Columns:** {file_info.get('cols')}")
        st.write(f"**Missing Fields:** {file_info.get('missing')}")

    if st.session_state.get("df") is not None and st.session_state.get("sample_info"):
        sample_info = st.session_state.get("sample_info")
        st.markdown("### 🧪 Sampling Summary")
        st.write(f"**Original Rows:** {sample_info.get('original_rows')}")
        st.write(f"**Sample Rows:** {sample_info.get('sample_rows')}")
        st.write(f"**Sample %:** {sample_info.get('sample_pct')}")
        st.write(f"**Sample Type:** {sample_info.get('sample_type')}")