"""OpenCR Report Module - Automated report generation."""

from opencr.report.annexA import AnnexAConfig, generate_annex_a
from opencr.report.metrics import generate_metrics_summary

__all__ = ["generate_annex_a", "AnnexAConfig", "generate_metrics_summary"]
