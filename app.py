"""
Streamlit App for SEG-D Seismic Data Loader using segdio
Supports SEG-D Rev 2.1 (24-bit and 32-bit data)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import tempfile
import os

try:
    from segdio import SEGD
except ImportError:
    st.error("segdio is not installed. Please install it using: pip install segdio")
    st.stop()

st.set_page_config(
    page_title="SEG-D Seismic Data Loader",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SEG-D Seismic Data Loader")
st.markdown("Upload a SEG-D file and explore its contents using `segdio` library.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload SEG-D File")
    uploaded_file = st.file_uploader(
        "Choose a SEG-D file",
        type=["segd", "sgd", "SEG-D", "bin"],
        help="Supported formats: SEG-D Rev 2.1 (24-bit and 32-bit)"
    )
    
    st.markdown("---")
    st.markdown("### About segdio")
    st.markdown(
        "`segdio` is a Python library for reading SEG-D seismic tape files. "
        "Currently supports SEG-D 2.1 (24 and 32 bits)."
    )
    st.markdown("[Documentation](https://github.com/geo-stack/segdio)")

# Helper functions
@st.cache_data
def load_segd(file_bytes):
    """Load SEG-D file from bytes and return SEGD object."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".segd") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        segd_obj = SEGD(tmp_path)
        return segd_obj, tmp_path
    except Exception as e:
        st.error(f"Error loading SEG-D file: {str(e)}")
        return None, None
    finally:
        os.unlink(tmp_path)  # Clean up temporary file

def get_file_summary(segd_obj):
    """Extract summary information from SEGD object."""
    summary = {
        "Number of channel sets": len(segd_obj.channel_set_headers),
        "Data types": {}
    }
    
    for i, cs in enumerate(segd_obj.channel_set_headers):
        num_traces = len(cs.trace_headers) if hasattr(cs, 'trace_headers') else 0
        summary["Data types"][f"Channel Set {i}"] = {
            "Number of traces": num_traces,
            "Trace headers available": num_traces > 0
        }
    
    return summary

def plot_trace(data, title="Seismic Trace"):
    """Create a plotly figure for a seismic trace."""
    if data is None or len(data) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        line=dict(color='blue', width=1),
        name='Amplitude'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400
    )
    return fig

def plot_wiggle(data, title="Wiggle Plot"):
    """Create a wiggle-style plot for seismic data."""
    if data is None or len(data) == 0:
        return None
    
    # Create wiggle plot by adding small offset per trace
    fig = go.Figure()
    
    # For now, plot as area fill for wiggle effect
    fig.add_trace(go.Scatter(
        y=data,
        fill='tozeroy',
        line=dict(color='black', width=1),
        name='Wiggle',
        fillcolor='rgba(0,0,255,0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400
    )
    return fig

# Main app logic
if uploaded_file is not None:
    with st.spinner("Loading SEG-D file..."):
        file_bytes = uploaded_file.read()
        segd_obj, _ = load_segd(file_bytes)
    
    if segd_obj is None:
        st.error("Failed to load SEG-D file. Please check the file format.")
        st.stop()
    
    # Display file information
    st.success(f"Successfully loaded: {uploaded_file.name}")
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 File Summary", "📈 Trace Viewer", "📊 Channel Sets"])
    
    with tab1:
        st.header("File Summary")
        summary = get_file_summary(segd_obj)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Channel Sets", summary["Number of channel sets"])
        
        with col2:
            st.metric("File Name", uploaded_file.name)
        
        st.subheader("Channel Set Information")
        for cs_name, cs_info in summary["Data types"].items():
            with st.expander(f"**{cs_name}**"):
                st.write(f"Number of traces: {cs_info['Number of traces']}")
                st.write(f"Trace headers available: {cs_info['Trace headers available']}")
    
    with tab2:
        st.header("Trace Viewer")
        
        if len(segd_obj.channel_set_headers) > 0:
            # Select channel set
            cs_indices = list(range(len(segd_obj.channel_set_headers)))
            selected_cs = st.selectbox(
                "Select Channel Set",
                cs_indices,
                format_func=lambda x: f"Channel Set {x}"
            )
            
            # Try to load trace data
            try:
                data = segd_obj.data(selected_cs)
                
                if data is not None and len(data) > 0:
                    st.success(f"Loaded {len(data)} samples from channel set {selected_cs}")
                    
                    # Display data statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min Amplitude", f"{np.min(data):.2f}")
                    with col2:
                        st.metric("Max Amplitude", f"{np.max(data):.2f}")
                    with col3:
                        st.metric("Mean Amplitude", f"{np.mean(data):.2f}")
                    
                    # Plot options
                    plot_type = st.radio("Plot Type", ["Standard Trace", "Wiggle Plot"], horizontal=True)
                    
                    if plot_type == "Standard Trace":
                        fig = plot_trace(data, title=f"Seismic Trace - Channel Set {selected_cs}")
                    else:
                        fig = plot_wiggle(data, title=f"Wiggle Plot - Channel Set {selected_cs}")
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download data option
                    csv_data = np.savetxt(BytesIO(), data, delimiter=",").getvalue().decode()
                    st.download_button(
                        label="Download Trace Data as CSV",
                        data=csv_data,
                        file_name=f"trace_cs{selected_cs}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data available for this channel set")
            
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        else:
            st.warning("No channel sets found in the file")
    
    with tab3:
        st.header("Channel Set Details")
        
        if len(segd_obj.channel_set_headers) > 0:
            for i, cs in enumerate(segd_obj.channel_set_headers):
                with st.expander(f"**Channel Set {i}**"):
                    # Display trace headers if available
                    if hasattr(cs, 'trace_headers') and len(cs.trace_headers) > 0:
                        st.subheader("Trace Headers")
                        # Show first few trace headers
                        num_headers = min(5, len(cs.trace_headers))
                        for j in range(num_headers):
                            st.write(f"Trace {j}: {cs.trace_headers[j]}")
                        
                        if len(cs.trace_headers) > 5:
                            st.info(f"... and {len(cs.trace_headers) - 5} more traces")
                    else:
                        st.info("No trace headers available for this channel set")
                    
                    # Additional channel set info
                    st.subheader("Channel Set Properties")
                    st.write(f"Type: {type(cs)}")
                    if hasattr(cs, 'data_format'):
                        st.write(f"Data format: {cs.data_format}")
        else:
            st.warning("No channel sets found")

else:
    # Show placeholder when no file is uploaded
    st.info("👈 Please upload a SEG-D file from the sidebar to begin")
    
    # Example usage instructions
    with st.expander("📖 How to use this app"):
        st.markdown("""
        1. **Upload a SEG-D file** using the sidebar file uploader
        2. **View file summary** in the first tab to understand the file structure
        3. **Explore trace data** in the second tab, including:
           - Select different channel sets
           - View standard trace plots or wiggle plots
           - Download trace data as CSV
        4. **Inspect channel set details** in the third tab
        
        **Supported formats:** SEG-D Revision 2.1 (24-bit and 32-bit data)
        
        **Note:** Large files may take some time to load. The app uses caching to improve performance.
        """)
