"""
SEG-D Seismic Plotter - Streamlit App with Manual Override
===========================================================
Upload a SEG-D (.sgd) file. If automatic header parsing fails, 
you can manually specify trace layout.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tempfile
import os
from io import BytesIO

# ------------------------------------------------------------
# 1. SEG-D reading functions (pysegd + fallback + manual)
# ------------------------------------------------------------

def read_segd_pysegd(filepath):
    """Read with pysegd library if available."""
    try:
        from pysegd import Segd
        seg = Segd(filepath)
    except ImportError:
        try:
            from pysegd import segd_read
            seg = segd_read(filepath)
        except ImportError:
            raise ImportError("pysegd not installed")
    except Exception as e:
        raise Exception(f"pysegd failed: {e}")

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
        padded[i, :len(t)] = t
    return padded, dt


def read_segd_manual(filepath, n_traces, n_samples, data_offset, byte_order='big', dt=0.002):
    """
    Read raw 32-bit floats from file.
    byte_order: 'big' (most SEG-D) or 'little'
    """
    with open(filepath, "rb") as f:
        raw = f.read()
    file_len = len(raw)
    bytes_per_trace = n_samples * 4
    expected_size = data_offset + n_traces * bytes_per_trace
    if expected_size > file_len:
        st.warning(f"File too short: need {expected_size} bytes, have {file_len}. Truncating traces.")
        max_possible = (file_len - data_offset) // bytes_per_trace
        if max_possible <= 0:
            raise ValueError("Data offset or sample count too large.")
        n_traces = max_possible
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    dtype = '>f4' if byte_order == 'big' else '<f4'
    for i in range(n_traces):
        start = data_offset + i * bytes_per_trace
        end = start + bytes_per_trace
        if end > file_len:
            break
        chunk = raw[start:end]
        if len(chunk) < bytes_per_trace:
            break
        traces[i] = np.frombuffer(chunk, dtype=dtype)
    return traces, dt


def auto_detect_or_manual(filepath, fallback_dt=None):
    """
    Try pysegd, then automatic fallback. If fallback fails, return None and
    provide instructions for manual override.
    """
    # First try pysegd
    try:
        import pysegd
        return read_segd_pysegd(filepath)
    except Exception as e:
        st.info(f"pysegd not available or failed: {e}. Trying built‑in auto‑detect...")
    
    # Automatic fallback (simplified, no crash)
    try:
        traces, dt = auto_read_segd(filepath)
        if traces is not None:
            if dt is None or dt <= 0:
                dt = fallback_dt or 0.002
            return traces, dt
    except Exception as e:
        st.warning(f"Auto‑detect failed: {e}")
    
    return None, None


def auto_read_segd(filepath):
    """
    A safer auto-detection that returns (None, None) instead of crashing.
    It tries common header layouts.
    """
    with open(filepath, "rb") as f:
        raw = f.read()
    file_len = len(raw)

    # Helper: BCD to int
    def bcd2int(b):
        return ((b >> 4) & 0x0F) * 10 + (b & 0x0F)

    def bcd2int16(hi, lo):
        return bcd2int(hi) * 100 + bcd2int(lo)

    # Try general header (32 bytes)
    if file_len < 32:
        return None, None

    # Sample interval from bytes 21-22 (BCD, units of 1/16 ms)
    base_si = bcd2int16(raw[21], raw[22]) if file_len > 22 else 0
    dt = base_si / 16.0 / 1000.0 if 0 < base_si < 10000 else None

    # Channel set header (32 bytes after general header)
    cs_start = 32
    if file_len < cs_start + 32:
        return None, None

    # Try to read n_traces and n_samples (as BCD or binary)
    n_traces_bcd = bcd2int16(raw[cs_start+6], raw[cs_start+7])
    n_traces_bin = int.from_bytes(raw[cs_start+6:cs_start+8], 'big')
    n_traces = n_traces_bcd if 1 <= n_traces_bcd <= 10000 else n_traces_bin
    if not (1 <= n_traces <= 10000):
        n_traces = 1

    n_samples_bcd = (bcd2int(raw[cs_start+8]) * 100 + bcd2int(raw[cs_start+9])) * 100
    n_samples_bin = int.from_bytes(raw[cs_start+8:cs_start+10], 'big')
    n_samples = n_samples_bcd if 10 <= n_samples_bcd <= 100000 else n_samples_bin
    if not (10 <= n_samples <= 50000):
        n_samples = 0

    # Compute offset: assume no extended headers for simplicity
    data_offset = 32 + 32  # general + one channel set

    # If n_samples still unknown, infer from file size
    if n_samples <= 0 and n_traces > 0:
        bytes_remaining = file_len - data_offset
        if bytes_remaining > 0:
            n_samples = bytes_remaining // (n_traces * 4)
            if n_samples <= 0:
                n_samples = bytes_remaining // 4  # assume one trace
                n_traces = 1
        else:
            return None, None

    bytes_per_trace = n_samples * 4
    max_traces = (file_len - data_offset) // bytes_per_trace
    if max_traces <= 0:
        return None, None
    if n_traces > max_traces:
        n_traces = max_traces

    # Read traces
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for i in range(n_traces):
        start = data_offset + i * bytes_per_trace
        end = start + bytes_per_trace
        if end > file_len:
            break
        traces[i] = np.frombuffer(raw[start:end], dtype=">f4")
    if dt is None or dt <= 0:
        dt = 0.002
    return traces, dt


# ------------------------------------------------------------
# 2. Plotting function (unchanged)
# ------------------------------------------------------------

def plot_seismic(traces, dt, clip_pct=98, max_traces=None,
                 title="Seismic Section", figsize=(12, 8)):
    """Generate a figure with wiggle traces over a heatmap."""
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
# 3. Streamlit UI with Manual Override
# ------------------------------------------------------------

st.set_page_config(page_title="SEG-D Seismic Plotter", layout="wide")
st.title("📈 SEG-D Seismic Plotter")
st.markdown("Upload a `.sgd` file – automatic header parsing; if it fails, you can manually specify the data layout.")

uploaded_file = st.file_uploader("Choose a SEG-D file (.sgd)", type=["sgd", "SGD"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sgd") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Try automatic detection first
    traces, dt = auto_detect_or_manual(tmp_path)

    if traces is not None:
        # Automatic succeeded – show sidebar with normal options
        st.success("✅ Automatic header parsing succeeded!")
        with st.expander("Show file info"):
            st.write(f"Traces: {traces.shape[0]}, Samples per trace: {traces.shape[1]}")
            st.write(f"Detected dt: {dt*1000:.3f} ms")
        use_manual = False
    else:
        st.error("❌ Automatic header parsing failed.")
        st.info("Please use the manual parameters below to read your file.")
        use_manual = True

    # Sidebar controls (common)
    st.sidebar.header("Plot Parameters")
    clip_pct = st.sidebar.slider("Amplitude Clip Percentile", 90, 100, 98, 1)
    max_traces = st.sidebar.number_input("Max Traces to Plot", min_value=1, value=500, step=50)
    if max_traces <= 0:
        max_traces = None

    if not use_manual:
        # Automatic mode – allow dt override
        dt_override = st.sidebar.number_input("Override dt (seconds) – 0 to keep detected",
                                              value=0.0, step=0.0005, format="%.4f")
        if dt_override > 0:
            dt = dt_override
    else:
        # Manual mode – ask for parameters
        st.sidebar.header("Manual Data Layout (required)")
        n_traces_manual = st.sidebar.number_input("Number of traces", min_value=1, value=100, step=10)
        n_samples_manual = st.sidebar.number_input("Samples per trace", min_value=1, value=500, step=50)
        data_offset_manual = st.sidebar.number_input("Data offset (bytes)", min_value=0, value=64, step=8,
                                                     help="Bytes to skip before data (e.g., 64 for SEG-D without extended headers)")
        byte_order = st.sidebar.selectbox("Byte order", ["big", "little"], index=0,
                                          help="Most SEG-D files are big-endian")
        dt_manual = st.sidebar.number_input("Sample interval (seconds)", min_value=0.0001, value=0.002, step=0.0005, format="%.4f")

        if st.sidebar.button("Load with these parameters"):
            try:
                traces, dt = read_segd_manual(tmp_path, n_traces_manual, n_samples_manual,
                                              data_offset_manual, byte_order, dt_manual)
                st.success(f"Loaded {traces.shape[0]} traces × {traces.shape[1]} samples")
                use_manual = False  # now we have data
            except Exception as e:
                st.error(f"Failed to load: {e}")
                traces = None

    # Plot if we have traces
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
        if use_manual:
            st.warning("Please adjust manual parameters and click 'Load with these parameters'.")
        else:
            st.warning("Could not load file automatically. Try manual mode.")

    os.unlink(tmp_path)

else:
    st.info("👈 Upload a SEG-D (.sgd) file to start.")
    st.markdown("""
    **Troubleshooting:**  
    If automatic detection fails, use **manual mode** (appears automatically).  
    Common values to try:
    - **Data offset**: 64 bytes (general header 32 + channel set header 32)  
    - **Byte order**: big-endian  
    - **Samples per trace**: try 500, 1000, 2000, or compute from file size: `(file_size - offset) / (4 * traces)`  
    - **Number of traces**: 1, 100, 200, or compute from file size
    """)
