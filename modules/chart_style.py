"""
chart_style.py
-----------------
Shared visual identity for every chart in the audit analytics toolkit.

WHY THIS EXISTS:
By Day 8 we had four chart-producing functions (Benford comparison,
outlier boxplot, reconciliation variances, risk distribution), each built
on the day its module was written — meaning each picked its own colors,
spacing, and spine styling independently. Individually they're fine, but
assembled into one PDF report (Week 2, Days 10-11) they'd look like five
unrelated documents stapled together. This module centralizes the visual
language so every chart shares one consistent identity, the way a real
audit firm's report template would.
"""

import matplotlib.pyplot as plt
from datetime import datetime

# Consistent risk/severity color language used across EVERY chart in the
# toolkit. Once a color means "Critical" in one chart, it means the same
# thing everywhere else — this consistency is what makes a multi-chart
# report feel coherent rather than assembled ad hoc.
COLORS = {
    "critical": "#7b241c",   # dark maroon — most severe (ghost loans, max risk)
    "high": "#c0392b",       # red — high risk / actual vs expected deviation
    "medium": "#e67e22",     # orange — medium risk / material variance
    "low": "#2c3e50",        # navy — low risk / baseline / expected values
    "neutral": "#95a5a6",    # grey — informational, non-finding
    "clean": "#1e8449",      # green — clean tie-outs, passed checks
    "accent": "#2980b9",     # blue — secondary accent where a fifth color is needed
}

FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (10, 5.5)
DPI = 150


def apply_style():
    """
    Sets shared matplotlib rcParams. Call once at the top of any plotting
    function in the toolkit, before creating a figure.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })


def add_footer(fig, source_text: str = "Amani Microfinance Ltd — Audit Analytics & Anomaly Detection System"):
    """
    Stamps a small, consistent attribution/timestamp footer on a chart —
    the kind of detail that makes exported visuals look report-ready
    rather than like a quick matplotlib output.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    fig.text(0.01, 0.005, f"{source_text}  |  Generated {timestamp}",
              fontsize=7, color="#888888", ha="left", va="bottom")
