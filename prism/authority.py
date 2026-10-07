"""Authority array helpers (Member C).

Citation acquisition and persistence are dataset/build concerns; these pure
functions turn aligned citation counts and publication months into g(d).
"""
from __future__ import annotations

import numpy as np


def authority_scores(citations, publish_months) -> tuple[np.ndarray, np.ndarray]:
    """Return raw log-scaled and month-cohort percentile authority arrays."""
    counts = np.asarray([0 if value is None else max(0, float(value)) for value in citations], dtype=np.float64)
    months = list(publish_months)
    if len(counts) != len(months):
        raise ValueError("citations and publish_months must have equal lengths")
    maximum = float(counts.max()) if counts.size else 0.0
    raw = np.log1p(counts) / np.log1p(maximum) if maximum > 0 else np.zeros_like(counts)

    labels = np.asarray([str(month) if month else "__unknown__" for month in months], dtype=object)
    unique, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    # Merge cohorts smaller than 30 into their publication year, as specified.
    for cohort_idx, cohort in enumerate(unique):
        if cohort == "__unknown__" or sizes[cohort_idx] >= 30:
            continue
        year = cohort[:4]
        labels[labels == cohort] = year if year else "__unknown__"

    cohort = np.zeros(len(counts), dtype=np.float64)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        values = counts[ids]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        # Minimum rank gives tied values (including zero citation papers) the
        # same lowest percentile, matching the spec's method="min" behavior.
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        ends = np.r_[starts[1:], len(ids)]
        for start, end in zip(starts, ends):
            pct = float(start + 1) / len(ids)
            cohort[ids[order[start:end]]] = pct

    unknown = labels == "__unknown__"
    if unknown.any() and len(counts):
        order = np.argsort(counts, kind="stable")
        sorted_values = counts[order]
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        ends = np.r_[starts[1:], len(counts)]
        global_pct = np.zeros(len(counts), dtype=np.float64)
        for start, end in zip(starts, ends):
            global_pct[order[start:end]] = float(start + 1) / len(counts)
        cohort[unknown] = global_pct[unknown]
    return np.clip(raw, 0.0, 1.0), np.clip(cohort, 0.0, 1.0)


def recency_scores(years, publish_months=None) -> np.ndarray:
    """Fallback g(d): newer publication months (or years) receive higher scores."""
    years = list(years)
    months = list(publish_months) if publish_months is not None else [None] * len(years)
    if len(years) != len(months):
        raise ValueError("years and publish_months must have equal lengths")
    values = np.zeros(len(years), dtype=np.float64)
    for i, (year, month) in enumerate(zip(years, months)):
        if year is None or int(year) <= 0:
            continue
        year_number = int(year)
        month_number = None
        if month:
            try:
                parsed_year, parsed_month = str(month).split("-", 1)
                if int(parsed_year) == year_number and 1 <= int(parsed_month) <= 12:
                    month_number = int(parsed_month)
            except (ValueError, TypeError):
                pass
        # For year-only dates, use midyear so they sort within the right year.
        values[i] = year_number * 12 + (month_number - 1 if month_number else 5.5)
    known = values > 0
    out = np.zeros(len(values), dtype=np.float64)
    if known.any():
        low, high = float(values[known].min()), float(values[known].max())
        out[known] = 1.0 if high == low else (values[known] - low) / (high - low)
    return out
