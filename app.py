"""
SEG-D Seismic Plotter – Streamlit App
=======================================
Supports multiple backends:
- pysegd3 (Rev 3 files)
- ObsPy (Rev 2 / 3 via read_segd)
- Manual override (any file with user-provided parameters)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tempfile
import os
from io import BytesIO

# ------------------------------------------------------------
# 1. SEG-D readers (multiple backends)
# ------------------------------------------------------------

def read_with_pysegd3(filepath):
    """Attempt to read using pysegd3 library."""
    try:
        from pysegd3.readsegd3 import read_segd_rev3
        traces_list = []
        dt = None
        for trace_header, trace_data in read_segd_rev3(filepath):
            traces_list.append(np.array(trace_data, dtype=np.float32))
            if dt is None:
                try:
                    dt = trace_header.sample_interval * 1e-6
                except:
                    pass
        if traces_list:
            traces = np.vstack(traces_list)
            if dt is None or dt <= 0:
                dt = 0.002
            return traces, dt
    except Exception as e:
        st.info(f"pysegd3 failed: {e}")
    return None, None


def read_with_obspy(filepath):
    """Attempt to read using ObsPy's SEG-D module (requires read_segd plugin)."""
    try:
        from obspy import read
        # Use a different variable name to avoid conflict with streamlit 'st'
        stream = read(filepath, format="SEGD")
        if len(stream) > 0:
            traces = np.vstack([tr.data for tr in stream])
            dt = stream[0].stats.delta if stream[0].stats.delta else 0.002
            return traces, dt
    except Exception as e:
        st.info(f"ObsPy SEG-D read failed: {e}")
    return None, None


def read_manual(filepath, n_traces, n_samples, data_offset, byte_order='big', dt=0.002):
    """Read raw 32-bit floats with user-provided layout."""
    with open(filepath, "rb") as f:
        raw = f.read()
    file_len = len(raw)
    bytes_per_trace = n_samples * 4
    expected_size = data_offset + n_traces * bytes_per_trace
    if expected_size > file_len:
        max_possible = (file_len - data_offset) // bytes_per_trace
        if max_possible <= 0:
            raise ValueError("Offset or sample count too large for file.")
        st.warning(f"Truncating traces from {n_traces} to {max_possible}")
        n_traces = max_possible
    dtype = '>f4' if byte_order == 'big' else '<f4'
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for i in range(n_traces):
        start = data_offset + i * bytes_per_trace
        end = start + bytes_per_trace
        if end > file_len:
            break
        traces[i] = np.frombuffer(raw[start:end], dtype=dtype)
    return traces, dt


def load_segd(filepath, fallback_dt=None, manual_params=None):
    """
    Try backends in order: pysegd3, ObsPy, then manual.
    manual_params = dict with keys: n_traces, n_samples, data_offset, byte_order, dt
    """
    # 1. pysegd3
    traces, dt = read_with_pysegd3(filepath)
    if traces is not None:
        st.success("✅ Loaded using pysegd3")
        return traces, dt

    # 2. ObsPy
    traces, dt = read_with_obspy(filepath)
    if traces is not None:
        st.success("✅ Loaded using ObsPy")
        return traces, dt

    # 3. Manual (if parameters supplied)
    if manual_params:
        try:
            traces, dt = read_manual(filepath,
                                     manual_params['n_traces'],
                                     manual_params['n_samples'],
                                     manual_params['data_offset'],
                                     manual_params.get('byte_order', 'big'),
                                     manual_params.get('dt', fallback_dt or 0.002))
            st.success("✅ Loaded using manual parameters")
            return traces, dt
        except Exception as e:
            st.error(f"Manual load failed: {e}")
    return None, None


# ------------------------------------------------------------
# 2. Plotting function
# ------------------------------------------------------------

def plot_seismic(traces, dt, clip_pct=98, max_traces=None,
                 title="Seismic Section", figsize=(12, 8)):
    if max_traces and traces.shape[0] > max_traces:
        traces = traces[:max_traces]

    n_traces, n_samples = traces.shape
    time_ms = np.arange(n_samples) * dt * 1000.0

    clip = np.percentile(np.abs(traces), clip_pct)
    if clip == 0:
        clip = 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    extent = [0.5, n_traces + 0.5, time_ms[-1], time_ms[0]]
    im = ax.imshow(traces.T, aspect="auto", extent=extent,
                   cmap="seismic", vmin=-clip, vmax=clip,
                   interpolation="bilinear", alpha=0.85)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label("Amplitude", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    max_amp = np.max(np.abs(traces))
    if max_amp == 0:
        max_amp = 1.0
    wiggle_scale = 0.42 / max_amp
    step = max(1, n_traces // 300)

    for i in range(0, n_traces, step):
        x_center = i + 1
        wig = traces[i] * wiggle_scale
        x = x_center + wig
        ax.plot(x, time_ms, color="black", linewidth=0.35, alpha=0.7, zorder=3)
        ax.fill_betweenx(time_ms, x_center, x, where=(x >= x_center),
                         color="black", alpha=0.75, linewidth=0, zorder=3)

    ax.set_xlim(0.5, n_traces + 0.5)
    ax.set_ylim(time_ms[-1], time_ms[0])
    ax.set_xlabel("Trace Number", color="white", fontsize=12)
    ax.set_ylabel("Two-Way Time (ms)", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))
    ax.grid(color="#333", linewidth=0.4, linestyle="--", alpha=0.5)

    ax.text(0.01, 0.01,
            f"Clip: {clip_pct}th pct  |  dt: {dt*1000:.3f} ms  |  {n_traces} traces",
            transform=ax.transAxes, color="#aaa", fontsize=8, va="bottom")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# 3. Streamlit UI with manual override
# ------------------------------------------------------------

st.set_page_config(page_title="SEG-D Seismic Plotter", layout="wide")
st.title("📈 SEG-D Seismic Plotter")
st.markdown("Upload a `.sgd` file – the app will try multiple reading backends. If all fail, you can specify the data layout manually.")

uploaded_file = st.file_uploader("Choose a SEG-D file (.sgd)", type=["sgd", "SGD"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sgd") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Sidebar controls
    st.sidebar.header("Plot Parameters")
    clip_pct = st.sidebar.slider("Amplitude Clip Percentile", 90, 100, 98, 1)
    max_traces = st.sidebar.number_input("Max Traces to Plot", min_value=1, value=500, step=50)
    if max_traces <= 0:
        max_traces = None

    # Automatic loading with backends
    traces, dt = load_segd(tmp_path, fallback_dt=0.002)

    if traces is not None:
        # Success – allow dt override
        dt_override = st.sidebar.number_input("Override dt (seconds) – 0 to keep detected",
                                              value=0.0, step=0.0005, format="%.4f")
        if dt_override > 0:
            dt = dt_override
        st.success(f"Loaded {traces.shape[0]} traces × {traces.shape[1]} samples")
        st.info(f"Sample interval: {dt*1000:.3f} ms | Record length: {traces.shape[1]*dt*1000:.1f} ms")
    else:
        st.error("❌ All automatic readers failed.")
        st.info("Please provide manual layout parameters below.")
        st.sidebar.header("Manual Data Layout")
        with st.sidebar.form("manual_form"):
            n_traces_manual = st.number_input("Number of traces", min_value=1, value=100, step=10)
            n_samples_manual = st.number_input("Samples per trace", min_value=1, value=500, step=50)
            data_offset_manual = st.number_input("Data offset (bytes)", min_value=0, value=64, step=8)
            byte_order = st.selectbox("Byte order", ["big", "little"], index=0)
            dt_manual = st.number_input("Sample interval (seconds)", min_value=0.0001, value=0.002, step=0.0005, format="%.4f")
            submitted = st.form_submit_button("Load with manual parameters")
        if submitted:
            manual_params = {
                'n_traces': n_traces_manual,
                'n_samples': n_samples_manual,
                'data_offset': data_offset_manual,
                'byte_order': byte_order,
                'dt': dt_manual
            }
            traces, dt = load_segd(tmp_path, fallback_dt=dt_manual, manual_params=manual_params)

    # Plot if we have data
    if traces is not None:
        fig = plot_seismic(traces, dt, clip_pct=clip_pct,
                           max_traces=max_traces if max_traces != 0 else None,
                           title=os.path.basename(uploaded_file.name),
                           figsize=(12, 8))
        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.download_button("📥 Download plot as PNG", data=buf, file_name="seismic_plot.png",
                           mime="image/png")
    else:
        st.warning("No data loaded. Please check manual parameters or install additional readers.")

    os.unlink(tmp_path)

else:
    st.info("👈 Upload a SEG-D (.sgd) file to start.")
    st.markdown("""
    **Supported backends (in order):**
    - **pysegd3** (install with `pip install pysegd3`) – best for Rev 3 files
    - **ObsPy** (install with `pip install obspy`) – may require additional plugin `read_segd`
    - **Manual override** – works for any file if you know the layout

    **For your specific file (`001001.sgd`):**  
    If automatic fails, try manual with:
    - Data offset = 64 (typical)
    - Byte order = big-endian
    - Experiment with traces and samples until the plot looks like seismic data.
    """)
