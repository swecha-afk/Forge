"""SHAP Visualization Utility for Frontend & Judge Verification Console.

Generates high-resolution matplotlib horizontal bar charts and base64 PNG images
from TreeSHAP feature attributions, ready for Streamlit st.pyplot() or REST API transport.
"""

import base64
import io
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "artifacts", ".mplcache"))

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend safe for server / background execution
import matplotlib.pyplot as plt

from app.schemas.swarm_schemas import ShapFeatureDriver, StandardizedRiskOutput


def format_feature_label(name: str, val: float) -> str:
    """Formats raw feature names into clear vernacular / judge labels."""
    mapping = {
        "humidity_3d_avg": f"3-Day Humidity ({val:.1f}%)",
        "temp_min_c": f"Night Temp ({val:.1f}°C)",
        "temp_max_c": f"Day Temp ({val:.1f}°C)",
        "rain_7d_mm": f"7-Day Rain ({val:.1f} mm)",
        "soil_moisture": f"Soil Moisture ({val:.2f} m³/m³)",
        "crop_id": f"Crop Identifier ({int(val)})",
    }
    return mapping.get(name, f"{name} ({val:.1f})")


def generate_shap_figure(
    drivers: list[ShapFeatureDriver],
    predicted_disease: str,
    probability: float,
) -> plt.Figure:
    """Renders an aesthetic horizontal bar chart of TreeSHAP feature attributions."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)

    # Sort drivers so largest absolute impact is on top
    sorted_drivers = sorted(drivers, key=lambda d: abs(d.shap_value), reverse=False)

    labels = [
        format_feature_label(d.feature_name, d.feature_value) for d in sorted_drivers
    ]
    values = [d.shap_value for d in sorted_drivers]
    colors = [
        "#e63946" if v > 0 else "#2a9d8f" for v in values
    ]  # Red = Risk Elevator, Teal = Risk Mitigator

    bars = ax.barh(
        labels, values, color=colors, height=0.55, edgecolor="none", zorder=3
    )

    # Vertical reference line at 0.0
    ax.axvline(0, color="#6c757d", linewidth=1.2, linestyle="--", alpha=0.8, zorder=2)

    # Add numeric value annotations next to bars
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        sign = "+" if val > 0 else ""
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.3f}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
            color="#212529",
        )

    clean_disease = predicted_disease.replace("_", " ").title()
    ax.set_title(
        f"TreeSHAP Attribution: {clean_disease} Risk ({probability * 100:.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#1d3557",
    )
    ax.set_xlabel(
        "← Mitigates Risk (Log-Odds)        |        Elevates Risk (Log-Odds) →",
        fontsize=10,
        color="#495057",
        labelpad=10,
    )

    # Styling clean-up
    ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ced4da")
    ax.spines["bottom"].set_color("#ced4da")
    ax.tick_params(colors="#495057", labelsize=10)

    plt.tight_layout()
    return fig


def generate_shap_base64_image(
    drivers: list[ShapFeatureDriver],
    predicted_disease: str,
    probability: float,
) -> str:
    """Renders the SHAP figure and encodes it as a base64 PNG data string."""
    fig = generate_shap_figure(drivers, predicted_disease, probability)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def save_shap_plot_to_file(
    risk_output: StandardizedRiskOutput,
    filepath: str = "artifacts/shap_attribution_plot.png",
) -> str:
    """Saves high-res SHAP plot PNG to disk."""
    fig = generate_shap_figure(
        drivers=risk_output.shap_top_drivers,
        predicted_disease=risk_output.predicted_disease,
        probability=risk_output.calibrated_probability,
    )
    fig.savefig(filepath, format="png", bbox_inches="tight")
    plt.close(fig)
    return filepath
