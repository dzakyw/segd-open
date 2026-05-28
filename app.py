import streamlit as st
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import segpy
from segpy.reader import create_reader
from segpy.trace_header import TraceHeaderRev1
import tempfile
import os

st.set_page_config(page_title="SEG-D Loader (segpy)", layout="wide")

st.title("📊 SEG-D Seismic Data Loader (using segpy)")
st.markdown("Pure Python implementation – no compilation required.")

uploaded_file = st.sidebar.file_uploader("Upload SEG-D file", type=["segd", "sgy", "bin"])

@st.cache_data
def load_segpy(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sgy") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        reader = create_reader(tmp_path)
        return reader, tmp_path
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None
    finally:
        os.unlink(tmp_path)

if uploaded_file:
    with st.spinner("Loading..."):
        reader, _ = load_segpy(uploaded_file.read())
    
    if reader:
        st.success(f"Loaded {uploaded_file.name}")
        st.metric("Number of traces", len(reader))
        
        trace_idx = st.slider("Select trace number", 0, len(reader)-1, 0)
        trace = reader.trace(trace_idx)
        
        fig = go.Figure(data=go.Scatter(y=trace, mode='lines'))
        fig.update_layout(title=f"Trace {trace_idx}", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: download
        csv = np.savetxt(BytesIO(), trace, delimiter=",").getvalue().decode()
        st.download_button("Download CSV", csv, f"trace_{trace_idx}.csv")
else:
    st.info("👈 Upload a SEG-Y or SEG-D file")
