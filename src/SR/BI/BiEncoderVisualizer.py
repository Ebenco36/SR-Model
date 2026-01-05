#!/usr/bin/env python3

"""
ULTRA-FIXED: BiEncoder Visualizer with all Plotly compatibility fixes.

All pandas Series converted to numpy arrays or lists BEFORE Plotly.
All range objects converted to lists.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class BiEncoderVisualizer:
    """Generate publication-quality visualizations for model comparison."""

    def __init__(self, output_dir: Path = Path("visualizations")):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def add_result(self, model_key: str, metrics: Dict[str, Any]) -> None:
        """Add evaluation results for a model."""
        self.results[model_key] = metrics

    def _create_df(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            raise ValueError("No results to visualize")
        return pd.DataFrame(self.results).T

    def plot_similarity_metrics(self) -> str:
        """Bar chart: Mean ± Std cosine similarity."""
        df = self._create_df()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df.index.tolist(),
            y=df["mean_similarity"].values.astype(float),
            error_y=dict(type="data", array=df["std_similarity"].values.astype(float)),
            name="Mean ± Std Similarity",
            marker=dict(color="lightblue", line=dict(color="darkblue", width=2)),
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.4f}<extra></extra>",
        ))

        fig.update_layout(
            title="Cosine Similarity on Test Set",
            xaxis_title="Model",
            yaxis_title="Cosine Similarity",
            template="plotly_white",
            font=dict(size=12),
            height=500,
            width=800,
            hovermode="x unified",
        )

        output_file = self.output_dir / "01_similarity_metrics.html"
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def plot_ranking_metrics(self) -> str:
        """Grouped bar chart: Recall@5, nDCG@5, MRR, MAP@5."""
        df = self._create_df()

        metrics = ["recall_at_5", "ndcg_at_5", "mrr", "map_at_5"]
        fig = go.Figure()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        model_names = df.index.tolist()

        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                x=model_names,
                y=df[metric].values.astype(float),
                name=metric.replace("_", "@").upper(),
                marker_color=colors[i],
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>",
            ))

        fig.update_layout(
            title="Ranking Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            font=dict(size=11),
            height=500,
            width=1000,
            hovermode="x unified",
        )

        output_file = self.output_dir / "02_ranking_metrics.html"
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def plot_efficiency_metrics(self) -> str:
        """Scatter plot: Model Size vs Inference Time (bubble = parameters)."""
        df = self._create_df()

        # Convert pandas Series to numpy arrays (FIX FOR PLOTLY)
        inference_times = df["inference_time_ms"].values.astype(float)
        model_sizes = df["model_size_mb"].values.astype(float)
        num_params = df["num_parameters"].values.astype(float) / 1e6  # Convert to millions
        similarities = df["mean_similarity"].values.astype(float)
        model_names = df.index.tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=inference_times,
            y=model_sizes,
            mode="markers+text",
            marker=dict(
                size=num_params * 0.5,  # Scale for visibility
                color=similarities,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Mean<br>Similarity"),
                line=dict(width=2, color="white"),
                sizemode="diameter",
                sizeref=2.0,
                sizemin=10,
            ),
            text=model_names,
            textposition="top center",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Inference: %{x:.2f} ms<br>"
                "Size: %{y:.1f} MB<br>"
                "Mean Similarity: %{marker.color:.4f}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title="Model Efficiency: Speed vs Size (colored by accuracy)",
            xaxis_title="Inference Time (ms/sentence)",
            yaxis_title="Model Size (MB)",
            template="plotly_white",
            font=dict(size=12),
            height=600,
            width=900,
            hovermode="closest",
        )

        output_file = self.output_dir / "03_efficiency_metrics.html"
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def plot_accuracy_vs_speed(self) -> str:
        """Scatter plot: Mean Similarity vs Inference Time."""
        df = self._create_df()

        # Convert to numpy arrays (FIX FOR PLOTLY)
        inference_times = df["inference_time_ms"].values.astype(float)
        similarities = df["mean_similarity"].values.astype(float)
        model_names = df.index.tolist()
        
        # Convert range to list (FIX FOR PLOTLY COLOR ISSUE)
        color_list = list(range(len(df)))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=inference_times,
            y=similarities,
            mode="markers+text",
            marker=dict(
                size=12,
                color=color_list,  # ← Convert range to list
                colorscale="Plotly3",
                showscale=False,
                line=dict(width=2, color="white"),
            ),
            text=model_names,
            textposition="top center",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Speed: %{x:.2f} ms/sentence<br>"
                "Accuracy: %{y:.4f}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title="Accuracy-Speed Trade-off",
            xaxis_title="Inference Time (ms/sentence) [Lower is Better]",
            yaxis_title="Mean Cosine Similarity [Higher is Better]",
            template="plotly_white",
            font=dict(size=12),
            height=600,
            width=900,
            hovermode="closest",
        )

        output_file = self.output_dir / "04_accuracy_vs_speed.html"
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def plot_metrics_heatmap(self) -> str:
        """Heatmap: All metrics normalized [0, 1] for comparison."""
        df = self._create_df()

        # Select key metrics
        metrics_cols = [
            "mean_similarity",
            "recall_at_5",
            "ndcg_at_5",
            "mrr",
            "map_at_5",
        ]

        heatmap_df = df[metrics_cols].copy()

        # Normalize to [0, 1]
        for col in heatmap_df.columns:
            min_val = heatmap_df[col].min()
            max_val = heatmap_df[col].max()
            if max_val > min_val:
                heatmap_df[col] = (heatmap_df[col] - min_val) / (max_val - min_val)
            else:
                heatmap_df[col] = 0.5

        # Convert to numpy for Plotly (FIX)
        z_values = heatmap_df.values.astype(float)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=[c.replace("_", " ").title() for c in heatmap_df.columns],
            y=heatmap_df.index.tolist(),
            colorscale="RdYlGn",
            text=np.round(z_values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Normalized Score"),
        ))

        fig.update_layout(
            title="Model Metrics Heatmap (Normalized to [0, 1])",
            xaxis_title="Metric",
            yaxis_title="Model",
            template="plotly_white",
            font=dict(size=11),
            height=500,
            width=900,
        )

        output_file = self.output_dir / "05_metrics_heatmap.html"
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def plot_summary_dashboard(self) -> str:
        """Multi-panel dashboard with key metrics."""
        df = self._create_df()

        # Convert to numpy arrays (FIX FOR PLOTLY)
        mean_sim = df["mean_similarity"].values.astype(float)
        recall_5 = df["recall_at_5"].values.astype(float)
        model_size = df["model_size_mb"].values.astype(float)
        infer_time = df["inference_time_ms"].values.astype(float)
        model_names = df.index.tolist()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Mean Similarity",
                "Recall@5",
                "Model Size (MB)",
                "Inference Time (ms)",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Mean Similarity
        fig.add_trace(
            go.Bar(x=model_names, y=mean_sim, name="Mean Sim",
                   marker_color="steelblue"),
            row=1, col=1
        )

        # Recall@5
        fig.add_trace(
            go.Bar(x=model_names, y=recall_5, name="Recall@5",
                   marker_color="coral"),
            row=1, col=2
        )

        # Model Size
        fig.add_trace(
            go.Bar(x=model_names, y=model_size, name="Size (MB)",
                   marker_color="lightgreen"),
            row=2, col=1
        )

        # Inference Time
        fig.add_trace(
            go.Bar(x=model_names, y=infer_time, name="Speed (ms)",
                   marker_color="salmon"),
            row=2, col=2
        )

        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="MB", row=2, col=1)
        fig.update_yaxes(title_text="ms", row=2, col=2)

        fig.update_layout(
            title_text="Model Comparison Dashboard",
            showlegend=False,
            height=700,
            width=1200,
            template="plotly_white",
            font=dict(size=11),
        )

        output_file = self.output_dir / "06_summary_dashboard.html"
        fig.write_html(str(output_file))
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def generate_all(self) -> List[str]:
        """Generate all visualizations."""
        logger.info("=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)

        outputs = [
            self.plot_similarity_metrics(),
            self.plot_ranking_metrics(),
            self.plot_efficiency_metrics(),
            self.plot_accuracy_vs_speed(),
            self.plot_metrics_heatmap(),
            self.plot_summary_dashboard(),
        ]

        return outputs

    def save_results_json(self) -> str:
        """Save raw results as JSON."""
        output_file = self.output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def save_results_csv(self) -> str:
        """Save results as CSV."""
        df = self._create_df()
        output_file = self.output_dir / "results.csv"
        df.to_csv(output_file)
        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)

    def generate_html_index(self) -> str:
        """Generate index HTML linking all visualizations."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>BiEncoder Model Comparison</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .links { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }
        .link-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #f9f9f9; }
        .link-card h3 { margin-top: 0; }
        .link-card a { display: inline-block; margin-top: 10px; padding: 8px 12px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
        .link-card a:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 BiEncoder Model Comparison Report</h1>
        <div class="links">
            <div class="link-card">
                <h3>Similarity Metrics</h3>
                <p>Mean ± Std cosine similarity comparison</p>
                <a href="01_similarity_metrics.html">View →</a>
            </div>
            <div class="link-card">
                <h3>Ranking Metrics</h3>
                <p>Recall@5, nDCG@5, MRR, MAP@5</p>
                <a href="02_ranking_metrics.html">View →</a>
            </div>
            <div class="link-card">
                <h3>Efficiency Metrics</h3>
                <p>Model size vs inference speed</p>
                <a href="03_efficiency_metrics.html">View →</a>
            </div>
            <div class="link-card">
                <h3>Accuracy-Speed Trade-off</h3>
                <p>Similarity vs inference time</p>
                <a href="04_accuracy_vs_speed.html">View →</a>
            </div>
            <div class="link-card">
                <h3>Metrics Heatmap</h3>
                <p>All metrics normalized [0, 1]</p>
                <a href="05_metrics_heatmap.html">View →</a>
            </div>
            <div class="link-card">
                <h3>Summary Dashboard</h3>
                <p>4-panel overview of all models</p>
                <a href="06_summary_dashboard.html">View →</a>
            </div>
        </div>
    </div>
</body>
</html>"""

        output_file = self.output_dir / "index.html"
        with open(output_file, "w") as f:
            f.write(html)

        logger.info(f"✓ Saved: {output_file}")
        return str(output_file)