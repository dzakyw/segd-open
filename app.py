"""
SEG-D Seismic Plotter - Streamlit App
======================================
Upload a SEG-D (.sgd) file, interactively adjust plot parameters,
and view the seismic section with wiggle + heatmap overlay.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tempfile
import os
from io import BytesIO

# ------------------------------------------------------------
# 1. SEG-D reading functions (same robust fallback as before)
# ------------------------------------------------------------

def read_segd_pysegd(filepath):
    """Read with pysegd library."""
    try:
        from pysegd import Segd
        seg = Segd(filepath)
    except ImportError:
        from pysegd import segd_read
        seg = segd_read(filepath)
    except Exception:
        raise ImportError("pysegd found but cannot read file")

    traces = []
    dt = None
    for trace in seg.traces:
        traces.append(np.array(trace.data, dtype=np.float32))
        if dt is None:
            try:
                dt = trace.channel_set.sample_interval * 1e-6
            except Exception:
                pass
    if not traces:
        raise ValueError("No traces found.")
    traces = np.vstack([t[np.newaxis, :] for t in traces])
    max_len = max(t.shape[0] for t in traces)
    padded = np.zeros((len(traces), max_len), dtype=np.float32)
    for i, t in enumerate(traces):
        padded[i, : len(t)] = t
    return padded, dt

def read_segd_fallback(filepath):
    """Robust hand-rolled SEG-D Rev 0/1 reader."""
    with open(filepath, "rb") as f:
        raw = f.read()

    def bcd2int(byte):
        return (byte >> 4) * 10 + (byte & 0x0F)
    def bcd2int16(hi, lo):
        return bcd2int(hi) * 100 + bcd2int(lo)

    # General Header
    si_bcd_hi = raw[21] if len(raw) > 21 else 0
    si_bcd_lo = raw[22] if len(raw) > 22 else 0
    base_si = bcd2int16(si_bcd_hi, si_bcd_lo)
    dt = base_si / 16.0 / 1000.0 if base_si > 0 else None

    n_channel_sets = raw[27] if len(raw) > 27 else 1
    n_ext = bcd2int16(raw[28], raw[29]) if len(raw) > 29 else 0
    n_exth = bcd2int16(raw[30], raw[31]) if len(raw) > 31 else 0

    # Channel Set Header (first set)
    cs_start = 32
    if len(raw) < cs_start + 32:
        raise ValueError("File too short for channel-set header")
    n_traces_bcd = bcd2int16(raw[cs_start+6], raw[cs_start+7])
    n_traces_bin = int.from_bytes(raw[cs_start+6:cs_start+8], 'big')
    n_traces = n_traces_bcd if 1 <= n_traces_bcd <= 10000 else n_traces_bin

    n_samples_bcd = (bcd2int(raw[cs_start+8]) * 100 + bcd2int(raw[cs_start+9])) * 100
    n_samples_bin = int.from_bytes(raw[cs_start+8:cs_start+10], 'big')
    n_samples = n_samples_bcd if 10 <= n_samples_bcd <= 100000 else n_samples_bin

    if n_traces <= 0 or n_traces > 10000:
        n_traces = 1
    if n_samples <= 0 or n_samples > 50000:
        n_samples = 0

    header_bytes = 32 + n_channel_sets * 32 + n_ext * 32 + n_exth * 32
    data_offset = header_bytes

    bytes_remaining = len(raw) - data_offset
    if n_samples <= 0 and n_traces > 0:
        n_samples = bytes_remaining // (n_traces * 4)
        if n_samples <= 0:
            raise ValueError("Cannot determine number of samples")

    bytes_per_trace = n_samples * 4
    if data_offset + n_traces * bytes_per_trace > len(raw):
        max_possible = bytes_remaining // bytes_per_trace
        if max_possible > 0:
            st.warning(f"Header says {n_traces} traces, but file contains {max_possible}. Truncating.")
            n_traces = max_possible
        else:
            raise ValueError("Data size inconsistent")

    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for i in range(n_traces):
        start = data_offset + i * bytes_per_trace
        end = start + bytes_per_trace
        chunk = raw[start:end]
        if len(chunk) < bytes_per_trace:
            break
        traces[i] = np.frombuffer(chunk, dtype=">f4")

    if dt is None or dt <= 0:
        dt = 0.002
    return traces, dt

def load_segd(filepath, fallback_dt=None):
    """Try pysegd, fallback to built-in reader."""
    try:
        import pysegd
        traces, dt = read_segd_pysegd(filepath)
    except Exception as e:
        st.info(f"pysegd not available or failed: {e}. Using built‑in reader.")
        traces, dt = read_segd_fallback(filepath)
    if dt is None or dt <= 0:
        dt = fallback_dt if fallback_dt else 0.002
    return traces, dt


# ------------------------------------------------------------
# 2. Plotting function (adapted for Streamlit)
# ------------------------------------------------------------

def plot_seismic_streamlit(traces, dt, clip_pct=98, max_traces=None,
                           title="Seismic Section", figsize=(12, 8)):
    """
    Generate a matplotlib figure with wiggle + heatmap.
    Returns the figure object.
    """
    if max_traces and traces.shape[0] > max_traces:
        traces = traces[:max_traces]

    n_traces, n_samples = traces.shape
    time_ms = np.arange(n_samples) * dt * 1000.0

    clip = np.percentile(np.abs(traces), clip_pct)
    if clip == 0:
        clip = 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Background heatmap
    extent = [0.5, n_traces + 0.5, time_ms[-1], time_ms[0]]
    im = ax.imshow(traces.T, aspect="auto", extent=extent,
                   cmap="seismic", vmin=-clip, vmax=clip,
                   interpolation="bilinear", alpha=0.85)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cbar.set_label("Amplitude", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Wiggle traces
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
# 3. Streamlit UI
# ------------------------------------------------------------

st.set_page_config(page_title="SEG-D Seismic Plotter", layout="wide")
st.title("📈 SEG-D Seismic Plotter")
st.markdown("Upload a `.sgd` file and interactively adjust the plot.")

# File uploader
uploaded_file = st.file_uploader("Choose a SEG-D file (.sgd)", type=["sgd", "SGD"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file (the reader expects a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sgd") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Parameters sidebar
        st.sidebar.header("Plot Parameters")
        clip_pct = st.sidebar.slider("Amplitude Clip Percentile", 90, 100, 98, 1)
        max_traces = st.sidebar.number_input("Max Traces to Plot", min_value=1, value=500,
                                             step=50, help="Leave large to plot all")
        if max_traces <= 0:
            max_traces = None

        dt_user = st.sidebar.number_input("Sample Interval (seconds) – override auto", 
                                          value=0.0, step=0.0005, format="%.4f",
                                          help="0 = use detected value")
        dt_override = dt_user if dt_user > 0 else None

        # Load data
        with st.spinner("Reading SEG-D file ..."):
            traces, dt = load_segd(tmp_path, fallback_dt=dt_override)
        if dt_override is not None:
            dt = dt_override

        st.success(f"Loaded {traces.shape[0]} traces × {traces.shape[1]} samples")
        st.info(f"Sample interval: {dt*1000:.3f} ms  |  Record length: {traces.shape[1]*dt*1000:.1f} ms")

        # Generate plot
        fig = plot_seismic_streamlit(traces, dt, clip_pct=clip_pct,
                                     max_traces=max_traces if max_traces != 0 else None,
                                     title=os.path.basename(uploaded_file.name),
                                     figsize=(12, 8))

        # Display
        st.pyplot(fig)

        # Optional: download button
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.download_button("Download plot as PNG", data=buf, file_name="seismic_plot.png",
                           mime="image/png")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)

    finally:
        # Clean up temp file
        os.unlink(tmp_path)
else:
    st.info("Please upload a SEG-D (.sgd) file to begin.")
    st.markdown("""
    **Example usage:**  
    - Upload a file (e.g., `001001.sgd`)  
    - Adjust clip percentile and max traces in the sidebar  
    - Optionally override the sample interval if header is missing  
    - View the seismic section with wiggle + colour overlay  
    - Download the plot as PNG
    """)
