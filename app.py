"""
Greenfield Surface Settlement Screening Tool
UQ CIVL4600 - Shallow Transport Tunnels in Residual Soils - Dean Blumson - Dr Jurij Karlovsek

Gaussian settlement trough: S(y) = Smax * exp(-y²/(2*i²))
where i = K * z0, with soil-type dependent K values
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Tuple, Dict
import os
IMG_PATH = os.path.join(os.path.dirname(__file__), "diagram.png")


# Soil type K-value ranges
SOIL_TYPES: Dict[str, Dict[str, float]] = {
    'SAND': {'k_lower': 0.25, 'k_upper': 0.45},
    'CLAY': {'k_lower': 0.40, 'k_upper': 0.60},
    'RESIDUAL SOIL': {'k_lower': 0.12, 'k_upper': 0.52}
}

N_POINTS = 800


def gaussian_settlement(y: np.ndarray, smax: float, i: float) -> np.ndarray:
    """
    Calculate Gaussian settlement profile.
    
    Args:
        y: Offset from tunnel centreline (m)
        smax: Maximum settlement at centreline (m)
        i: Trough width parameter (m)
    
    Returns:
        Settlement S(y) (m)
    """
    return smax * np.exp(-y**2 / (2 * i**2))


def make_dataframe(y: np.ndarray, s_lower: np.ndarray, s_upper: np.ndarray) -> pd.DataFrame:
    """
    Create DataFrame with settlement profile data.
    
    Args:
        y: Offset array (m)
        s_lower: Lower envelope settlement (m)
        s_upper: Upper envelope settlement (m)
    
    Returns:
        DataFrame with columns [y, S_lower, S_upper]
    """
    return pd.DataFrame({
        'y (m)': y,
        'S_lower (m)': s_lower,
        'S_upper (m)': s_upper
    })


def make_plot(y: np.ndarray, s_lower: np.ndarray, s_upper: np.ndarray, 
              k_lower: float, k_upper: float) -> plt.Figure:
    """
    Create matplotlib figure with settlement profiles.
    
    Args:
        y: Offset array (m)
        s_lower: Lower envelope settlement (m)
        s_upper: Upper envelope settlement (m)
        k_lower: Lower K value
        k_upper: Upper K value
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with inverted y-axis (settlement is positive downward)
    ax.plot(y, s_lower, 'b-', linewidth=2, label=f'Lower (K={k_lower})')
    ax.plot(y, s_upper, 'r-', linewidth=2, label=f'Upper (K={k_upper})')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlabel('Offset y (m)', fontsize=12)
    ax.set_ylabel('Settlement S(y) (m)', fontsize=12)
    ax.set_title('Greenfield Surface Settlement Profile', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    
    # Invert y-axis so settlement (positive values) appears downward
    ax.invert_yaxis()
    
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """
    Convert matplotlib figure to PNG bytes for download.
    
    Args:
        fig: Matplotlib figure
    
    Returns:
        PNG image as bytes
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def main():
    st.set_page_config(
        page_title="Tunnel Settlement Screening",
        page_icon="🚇",
        layout="wide"
    )
    
    # Main panel header and description
    st.title("🚇 Greenfield Surface Settlement Screening Tool")
    st.markdown("**UQ CIVL4600 - Shallow Transport Tunnels in Residual Soils - Dean Blumson - Dr Jurij Karlovsek**")
    
    st.markdown("""
    This tool applies the Gaussian settlement trough model
    """)
    
    st.latex(r"S(y) = S_{\max} \, e^{-\frac{y^2}{2i^2}}, \quad \text{where } i = K \, z_0")
    
    st.markdown("""
    **Parameter Definitions:**
    
    - *S(y)* is the settlement of the tunnel at distance *y* along the boundary
    - *S*<sub>max</sub> is the maximum settlement at the tunnel centreline
    - *y* is the offset distance from the tunnel centreline
    - *i* is the trough width parameter
    - *K* is an empirical coefficient dependent on soil type
    - *z*<sub>0</sub> is the depth to the tunnel axis
    
    to estimate greenfield surface settlement above shallow transport tunnels, as illustrated below by Khoo, Idris, Mohamad & Rashid in 2018.
    """, unsafe_allow_html=True)
    
    # Display diagram image centered at 62.5% width (50% * 1.25)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(IMG_PATH,
                 caption="Settlement trough of Gaussian form (Khoo, Idris, Mohamad & Rashid, 2018)",
                 use_column_width=True)
    
    st.markdown("""
    Select the relevant soil type to update the empirical K-range used in computing the lower and upper settlement envelopes:
    
    | Soil Type | K<sub>lower</sub> | K<sub>upper</sub> | Notes |
    |-----------|-------------------|-------------------|-------|
    | **Sand** | 0.25 | 0.45 | Narrower, depth-controlled troughs |
    | **Clay** | 0.40 | 0.60 | Wider, smoother profiles typical of cohesive ground |
    | **Residual Soil** | 0.12 | 0.52 | Variable response calibrated from SEQ case data |
    
    These bounds represent conservative screening envelopes derived from published case histories. Adjusting tunnel geometry or selecting a different soil type updates *i = Kz*<sub>0</sub> and re-plots both envelopes in real time, allowing visual comparison of trough width sensitivity across materials.
    """, unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Soil type selection
    st.sidebar.subheader("Soil Type")
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        sand_btn = st.button("SAND", use_container_width=True)
    with col2:
        clay_btn = st.button("CLAY", use_container_width=True)
    with col3:
        residual_btn = st.button("RESIDUAL SOIL", use_container_width=True)
    
    # Initialize or update soil type in session state
    if 'soil_type' not in st.session_state:
        st.session_state.soil_type = 'RESIDUAL SOIL'
    
    if sand_btn:
        st.session_state.soil_type = 'SAND'
    elif clay_btn:
        st.session_state.soil_type = 'CLAY'
    elif residual_btn:
        st.session_state.soil_type = 'RESIDUAL SOIL'
    
    # Display current soil type
    st.sidebar.info(f"**Current Soil Type:** {st.session_state.soil_type}")
    
    # Get K values for selected soil type
    k_lower = SOIL_TYPES[st.session_state.soil_type]['k_lower']
    k_upper = SOIL_TYPES[st.session_state.soil_type]['k_upper']
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tunnel Geometry")
    
    d = st.sidebar.number_input(
        "Tunnel Diameter D (m)",
        min_value=0.1,
        value=6.0,
        step=0.1,
        format="%.1f"
    )
    
    z0 = st.sidebar.number_input(
        "Depth to Tunnel Axis z₀ (m)",
        min_value=0.5,
        value=10.0,
        step=0.5,
        format="%.1f"
    )
    
    smax = st.sidebar.number_input(
        "Maximum Settlement S_max (m)",
        min_value=0.0,
        value=0.012,
        step=0.001,
        format="%.3f"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Plot Settings")
    
    default_y_max = max(30.0, 5.0 * d)
    y_max = st.sidebar.number_input(
        "Plot Half-Width Y_max (m)",
        min_value=5.0,
        value=default_y_max,
        step=1.0,
        format="%.1f"
    )
    
    # Validation
    if d <= 0 or z0 <= 0:
        st.error("❌ Error: Tunnel diameter D and depth z₀ must be greater than zero.")
        st.stop()
    
    if smax < 0:
        st.error("❌ Error: Maximum settlement S_max cannot be negative.")
        st.stop()
    
    # Coerce to minimum with warning
    if y_max < 5:
        st.warning("⚠️ Plot half-width coerced to minimum of 5 m.")
        y_max = 5.0
    
    # Calculations
    i_lower = k_lower * z0
    i_upper = k_upper * z0
    
    y = np.linspace(-y_max, y_max, N_POINTS)
    s_lower = gaussian_settlement(y, smax, i_lower)
    s_upper = gaussian_settlement(y, smax, i_upper)
    
    # Metrics
    st.subheader("Derived Parameters")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("i_lower (m)", f"{i_lower:.2f}")
    with col2:
        st.metric("i_upper (m)", f"{i_upper:.2f}")
    with col3:
        st.metric("i_lower/D (−)", f"{i_lower/d:.3f}")
    with col4:
        st.metric("i_upper/D (−)", f"{i_upper/d:.3f}")
    with col5:
        st.metric("S_max/D (−)", f"{smax/d:.6f}")
    
    # Plot
    st.subheader("Settlement Profile")
    fig = make_plot(y, s_lower, s_upper, k_lower, k_upper)
    st.pyplot(fig)
    
    # Downloads
    st.subheader("Download Results")
    col_csv, col_png = st.columns(2)
    
    df = make_dataframe(y, s_lower, s_upper)
    csv_data = df.to_csv(index=False).encode('utf-8')
    
    with col_csv:
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name="settlement_profile.csv",
            mime="text/csv"
        )
    
    with col_png:
        png_data = fig_to_png_bytes(fig)
        st.download_button(
            label="🖼️ Download PNG",
            data=png_data,
            file_name="settlement_profile.png",
            mime="image/png"
        )
    
    plt.close(fig)


if __name__ == "__main__":

    main()


