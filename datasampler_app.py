import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from filehub_app import upload_dataframe
from filehub_app import download_dataframe

# Sanity check for AWS credentials (for Streamlit Cloud)
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", os.getenv("S3_BUCKET_NAME"))
S3_REGION = st.secrets.get("S3_REGION", os.getenv("S3_REGION", "us-east-1"))

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME, S3_REGION]):
    st.error("Missing AWS credentials or S3 configuration. Please check your Streamlit secrets.")
    st.stop()

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
if uploaded_file is None and st.session_state.get("df") is not None and st.session_state.get("uploaded_file") is not None:
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

            if df.columns[-1] == "DS_SAMPLE":
                sample_val = df["DS_SAMPLE"].iloc[0]
                if sample_val.startswith("Sampled["):
                    try:
                        parts = sample_val.strip("Sampled[]").split("][")
                        sample_type = parts[0]
                        sample_counts = parts[1].split("/")
                        sample_rows = int(sample_counts[0])
                        original_rows = int(sample_counts[1])
                        pct = round(sample_rows / original_rows * 100, 2)
                        st.warning(
                            f"**⚠️ Sampled Data Detected**\n\n"
                            f"- Sample type: {sample_type}\n"
                            f"- Original size: {original_rows}\n"
                            f"- Sample size: {sample_rows}\n"
                            f"- Coverage: {pct}%"
                        )
                    except Exception as e:
                        st.info("Note: A 'DS_SAMPLE' column was found but could not be parsed.")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
else:
    st.session_state.uploaded_file = None

if st.session_state.get("df") is not None and st.session_state.get("file_info"):
    df = st.session_state.get("df")
    row_count = df.shape[0]
else:
    df = None

if st.session_state.pop("filehub_success", False):
    st.success("✅ File loaded from FileHub successfully!")

if df is not None:
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

                # Add DS_SAMPLE column (keeping previous as history)
                sampled_df = sampled_df.copy()
                existing_sample_cols = [col for col in sampled_df.columns if col.startswith("DS_SAMPLE")]

                for i, col in enumerate(existing_sample_cols, 1):
                    sampled_df.rename(columns={col: f"DS_SAMPLE_{i}"}, inplace=True)

                tag = f"Sampled[{sample_type}][{len(sampled_df)}/{row_count}]"
                sampled_df["DS_SAMPLE"] = tag

                # Ensure DS_SAMPLE is the last column
                ds_cols = [col for col in sampled_df.columns if col.startswith("DS_SAMPLE")]
                other_cols = [col for col in sampled_df.columns if not col.startswith("DS_SAMPLE")]
                sampled_df = sampled_df[other_cols + ds_cols]

                # Update session
                st.session_state.sampled_df = sampled_df
                st.session_state.sample_info = {
                    "original_rows": row_count,
                    "sample_rows": len(sampled_df),
                    "sample_pct": round(len(sampled_df) / row_count * 100, 2),
                    "sample_type": sample_type
                }

                st.success("Sampling completed successfully!")

            except Exception as e:
                st.error(f"Sampling failed: {e}")

if st.session_state.get("sampled_df") is not None:
    sampled_df = st.session_state.get("sampled_df")
    st.dataframe(sampled_df.head(20), use_container_width=True)

    if st.button("📤 Send to FileHub"):
        try:
            st.warning("📡 Uploading to FileHub...")
            token = upload_dataframe(sampled_df, source_app="DataSampler", original_filename="sampled_output.csv")
            st.success(f"✅ File sent to FileHub! Transfer Token: `{token}`")
            st.info("You can now use this token in DataWizard, DataBlender, or any other app.")
        except Exception as e:
            st.error("❌ Upload to FileHub failed.")
            st.exception(e)

# Sidebar (render at end after session state is set)
with st.sidebar:
    st.markdown("## 🔁 FileHub")
    transfer_token = st.text_input("Enter transfer token to load file from FileHub")
    if st.button("Submit token") and transfer_token.strip():
        try:
            df_from_filehub, original_filename = download_dataframe(transfer_token.strip())
            st.session_state.df = df_from_filehub
            st.session_state.uploaded_file = None
            st.session_state.sampled_df = None
            st.session_state.sample_info = {}

            st.session_state.file_info = {
                "name": original_filename,
                "size_mb": round(df_from_filehub.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "rows": df_from_filehub.shape[0],
                "cols": df_from_filehub.shape[1],
                "missing": df_from_filehub.isnull().sum().sum()
            }

            st.session_state.filehub_success = True
            st.rerun()
        except Exception as e:
            st.error("❌ Error retrieving file from FileHub.")
            st.exception(e)

    if st.session_state.get("df") is not None:
        cols = st.session_state.df.columns
        if cols[-1] == "DS_SAMPLE":
            sample_val = st.session_state.df["DS_SAMPLE"].iloc[0]
            if sample_val.startswith("Sampled["):
                try:
                    parts = sample_val.strip("Sampled[]").split("][")
                    sample_type = parts[0]
                    sample_counts = parts[1].split("/")
                    sample_rows = int(sample_counts[0])
                    original_rows = int(sample_counts[1])
                    pct = round(sample_rows / original_rows * 100, 2)
                    st.warning(
                        f"**⚠️ Sampled Data Detected**\n\n"
                        f"- Sample type: {sample_type}\n"
                        f"- Original size: {original_rows}\n"
                        f"- Sample size: {sample_rows}\n"
                        f"- Coverage: {pct}%"
                    )
                except Exception as e:
                    st.info("Note: A 'DS_SAMPLE' column was found but could not be parsed.")

    if st.session_state.get("df") is not None and st.session_state.get("file_info"):
        file_info = st.session_state.get("file_info")
        st.markdown("### 📁 File Summary")
        st.write(f"**Name:** {file_info.get('name')}")
        st.write(f"**Size:** {file_info.get('size_mb')} MB")
        st.write(f"**Rows:** {file_info.get('rows')}")
        st.write(f"**Columns:** {file_info.get('cols')}")
        st.write(f"**Missing Fields:** {file_info.get('missing')}")

    if st.session_state.get("sample_info"):
        st.markdown("## 📊 SampleStats")
        sample_info = st.session_state.get("sample_info")
        st.markdown("### 🧪 Sampling Summary")
        st.write(f"**Original Rows:** {sample_info.get('original_rows')}")
        st.write(f"**Sample Rows:** {sample_info.get('sample_rows')}")
        st.write(f"**Sample %:** {sample_info.get('sample_pct')}")
        st.write(f"**Sample Type:** {sample_info.get('sample_type')}")
