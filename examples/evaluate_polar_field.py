"""Plot and summarize the magnetic field in a saved polar-crop inversion.

Run from the checkout: python examples/evaluate_polar_field.py --help
Statistics use equally weighted, finite pixels in the saved crop after masking.
They describe the model's azimuth branch, not a disambiguated polar-cap flux.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.colors import Normalize

from spinnme import load
from spinnme.evaluation import evaluate_result, write_metrics
from spinnme.plotting import plot_maps, save_figure

PARAMETERS = ["Br", "Btheta", "Bphi", "Blos", "B"]


def training_observations(observations):
    """Select each saved training grid once, excluding validation duplicates."""
    selected = [row for row in observations if row.get("role") == "train"]
    selected = selected or list(observations)
    unique = {row["id"]: row for row in selected}
    if not unique:
        raise ValueError("The snapshot has no saved observation grids.")
    return sorted(unique.values(), key=lambda row: Time(row["time"]["model_time"]).utc.jd)


def summarize(result):
    """Extend the shared finite-pixel summaries with median and mean |B|."""
    report = evaluate_result(result)
    for row in report["metrics"]:
        values = np.asarray(result.arrays[row["parameter"]])
        valid = result.mask & np.isfinite(values)
        values = values[valid].astype(np.float64)
        row["median"] = float(np.median(values))
        row["mean_absolute"] = float(np.mean(np.abs(values)))
    return report


def plot_distributions(result, title):
    figure, axes = plt.subplots(1, len(PARAMETERS), figsize=(14, 3.2), layout="constrained")
    for axis, name in zip(axes, PARAMETERS):
        values = np.asarray(result.arrays[name])
        values = values[result.mask & np.isfinite(values)]
        axis.hist(values, bins=80, histtype="stepfilled", alpha=0.7)
        if name != "B":
            axis.axvline(0, color="black", linewidth=0.6)
        axis.set(xlabel=f"{name} [{result.units[name]}]", ylabel="Pixels")
    figure.suptitle(title)
    return figure


def plot_time_series(rows, title):
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    for parameter in ("Br", "Blos"):
        selected = [row for row in rows if row["parameter"] == parameter]
        dates = Time([row["time_utc"] for row in selected], scale="utc").to_datetime()
        axes[0].plot(dates, [row["mean"] for row in selected], "o-", label=parameter)
        axes[1].plot(dates, [row["mean_absolute"] for row in selected], "o-", label=parameter)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("Pixel mean field [G]")
    axes[1].set(ylabel="Pixel mean |field| [G]", xlabel="Observation time [UTC]")
    locator = mdates.AutoDateLocator()
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    return figure


def evaluate_polar_field(snapshot_path, output, *, label="Polar crop", device=None,
                         batch_size=512, mu_limit=0.0, field_limit=300.0, overwrite=False):
    """Write maps, histograms and a time series without retraining the model."""
    if not np.isfinite(field_limit) or field_limit <= 0:
        raise ValueError("field_limit must be finite and positive.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not np.isfinite(mu_limit) or not 0 <= mu_limit < 1:
        raise ValueError("mu_limit must be finite and in [0, 1).")
    output = Path(output).expanduser()
    snapshot_path = Path(snapshot_path).expanduser()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} is not empty; choose a new output or pass --overwrite.")
    rows = []
    metadata = {
        "snapshot": str(snapshot_path.resolve()), "label": label, "mu_limit": mu_limit,
        "weighting": "equal weight per finite pixel within each saved crop",
        "selection": "saved training observations if present; otherwise all saved grids",
        "field_limit_G": field_limit,
        "interpretation": "Model azimuth branch; no disambiguation, area weighting or reference comparison.",
    }
    with load(snapshot_path, device=device) as snapshot:
        observations = training_observations(snapshot.observations)
        for index, observation in enumerate(observations):
            print(f"[{index + 1}/{len(observations)}] {label}: {observation['id']}", flush=True)
            result = snapshot.evaluate(observation, parameters=PARAMETERS, batch_size=batch_size,
                                       mu_limit=mu_limit, progress=True)
            report = summarize(result)
            directory = output / f"observation_{index:03d}"
            epoch = result.metadata["model_time"]
            report["metadata"] = {**result.metadata, "summary": metadata}
            write_metrics(report, directory / "metrics.json")
            rows.extend({"observation_id": observation["id"], "time_utc": epoch,
                         "directory": directory.name, **row} for row in report["metrics"])

            signed = {name: Normalize(-field_limit, field_limit) for name in PARAMETERS if name != "B"}
            cmaps = {name: "RdBu_r" for name in signed}
            title = f"{label} — {epoch} UTC"
            figure = plot_maps(result, parameters=PARAMETERS, norms=signed, cmaps=cmaps,
                               source_label="SPINNME", title=title)
            for axis in figure.axes:
                if hasattr(axis, "coords"):
                    for coordinate in axis.coords:
                        coordinate.set_ticklabel(size=8, exclude_overlapping=True)
            save_figure(figure, directory / "magnetic_maps", dpi=300)
            plt.close(figure)
            figure = plot_distributions(result, title)
            save_figure(figure, directory / "distributions", dpi=300)
            plt.close(figure)

    write_metrics({"metadata": metadata, "metrics": rows}, output / "summary.json")
    figure = plot_time_series(rows, f"{label}: crop pixel averages")
    save_figure(figure, output / "time_series", dpi=300)
    plt.close(figure)
    print(f"Plots and summaries: {output.resolve()}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Saved inversion.pme")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Polar crop", help="Figure label, e.g. North pole")
    parser.add_argument("--device", default=None, help="cpu/cuda/cuda:N; default prefers CUDA if available")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mu-limit", type=float, default=0.0, help="Exclude pixels at or below this viewing cosine")
    parser.add_argument("--field-limit", type=float, default=300.0, help="Signed map color range is ±this many gauss")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate plots and summaries in an existing output")
    args = parser.parse_args(argv)
    evaluate_polar_field(args.snapshot, args.output, label=args.label, device=args.device,
                         batch_size=args.batch_size, mu_limit=args.mu_limit,
                         field_limit=args.field_limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
