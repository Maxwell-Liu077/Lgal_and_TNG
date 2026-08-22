"""Public plotting API assembled from focused, reusable plot modules."""

from .comparison import (
    INNER_HOT_DELIVERY_SERIES,
    REHEATING_SERIES,
    SUPPLY_COMPARISON_SERIES,
    plot_inner_hot_delivery_components,
    plot_reheating_comparison,
    plot_supply_comparison,
)
from .scaling_relation import (
    BASELINE_NON_SAM_SERIES,
    BASELINE_SERIES,
    INWARD_DELIVERY_NON_SAM_SERIES,
    INWARD_DELIVERY_SERIES,
    NON_SAM_SERIES,
    NORMALIZATION_STYLES,
    SERIES,
    _plot_selected_mass_dependence,
    plot_mass_dependence,
    plot_mass_dependence_without_sam,
    plot_normalized_mass_dependence,
)
from .style import ACADEMIC_STYLE, _save, save_figure
from .tracer_history import (
    HISTORY_BAR_COLORS,
    HISTORY_LABELS,
    plot_tracer_history_heatmaps,
    plot_tracer_history_top4_stacked_bars,
)

__all__ = [
    "ACADEMIC_STYLE",
    "BASELINE_NON_SAM_SERIES",
    "BASELINE_SERIES",
    "INNER_HOT_DELIVERY_SERIES",
    "INWARD_DELIVERY_NON_SAM_SERIES",
    "INWARD_DELIVERY_SERIES",
    "NON_SAM_SERIES",
    "NORMALIZATION_STYLES",
    "REHEATING_SERIES",
    "SERIES",
    "SUPPLY_COMPARISON_SERIES",
    "HISTORY_BAR_COLORS",
    "HISTORY_LABELS",
    "_plot_selected_mass_dependence",
    "_save",
    "plot_inner_hot_delivery_components",
    "plot_mass_dependence",
    "plot_mass_dependence_without_sam",
    "plot_normalized_mass_dependence",
    "plot_reheating_comparison",
    "plot_supply_comparison",
    "plot_tracer_history_heatmaps",
    "plot_tracer_history_top4_stacked_bars",
    "save_figure",
]
