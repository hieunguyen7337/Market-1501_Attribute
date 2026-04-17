from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass
class ModelRecord:
    name: str
    released_at: datetime
    context_tokens: int
    input_price_per_million: float
    output_price_per_million: float


CONTEXT_SUFFIXES = {
    "K": 1_000,
    "M": 1_000_000,
}
DEFAULT_MAX_OUTPUT_PRICE_PER_MILLION = 50.0


def parse_compact_number(text: str) -> int:
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([KM])", text.strip())
    if not match:
        raise ValueError(f"Unsupported context value: {text!r}")
    value, suffix = match.groups()
    return int(float(value) * CONTEXT_SUFFIXES[suffix])


def parse_price_line(text: str) -> float:
    match = re.search(r"\$([0-9]*\.?[0-9]+)\s*/M", text)
    if not match:
        raise ValueError(f"Unsupported price line: {text!r}")
    return float(match.group(1))


def parse_records(path: Path, max_output_price_per_million: float | None) -> list[ModelRecord]:
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in raw_lines if line]

    if len(lines) % 5 != 0:
        raise ValueError(
            f"Expected 5 non-empty lines per record in {path}, found {len(lines)} lines."
        )

    records: list[ModelRecord] = []
    for index in range(0, len(lines), 5):
        name, release_line, context_line, input_line, output_line = lines[index : index + 5]
        context_value = context_line.split()[0]
        record = ModelRecord(
            name=name,
            released_at=datetime.strptime(release_line, "%b %d, %Y"),
            context_tokens=parse_compact_number(context_value),
            input_price_per_million=parse_price_line(input_line),
            output_price_per_million=parse_price_line(output_line),
        )
        if (
            max_output_price_per_million is None
            or record.output_price_per_million <= max_output_price_per_million
        ):
            records.append(record)

    return sorted(records, key=lambda record: (record.released_at, record.name.lower()))


def attach_interaction(
    figure: Figure,
    axes: list[Axes],
    records: list[ModelRecord],
    x_positions: list[int],
    date_labels: list[str],
    output_prices: list[float],
    input_prices: list[float],
) -> None:
    y_series = [output_prices, input_prices]
    marker_artists = []
    crosshairs = [
        axis.axvline(
            x=x_positions[0],
            color="#444444",
            linestyle="--",
            linewidth=0.9,
            alpha=0.35,
            visible=False,
        )
        for axis in axes
    ]
    tooltip = axes[0].annotate(
        "",
        xy=(x_positions[0], output_prices[0]),
        xytext=(0.02, 0.98),
        textcoords="axes fraction",
        bbox={"boxstyle": "round", "fc": "white", "ec": "#666666", "alpha": 0.96},
        fontsize=9,
        zorder=20,
        annotation_clip=False,
        va="top",
        ha="left",
    )
    tooltip.set_visible(False)

    for axis, series in zip(axes, y_series):
        markers = axis.scatter(
            x_positions,
            series,
            s=48,
            alpha=0,
            picker=True,
        )
        markers.set_pickradius(10)
        marker_artists.append(markers)

    def hide_guides() -> None:
        for crosshair in crosshairs:
            crosshair.set_visible(False)
        tooltip.set_visible(False)
        figure.canvas.draw_idle()

    def on_move(event) -> None:
        if event.inaxes not in axes:
            hide_guides()
            return

        hovered_index = None
        for markers in marker_artists:
            contains, details = markers.contains(event)
            indices = details.get("ind")
            if contains and indices is not None and len(indices) > 0:
                hovered_index = int(indices[0])
                break

        if hovered_index is None:
            hide_guides()
            return

        record = records[hovered_index]
        x_position = x_positions[hovered_index]

        for crosshair in crosshairs:
            crosshair.set_xdata([x_position, x_position])
            crosshair.set_visible(True)

        tooltip.xy = (x_position, output_prices[hovered_index])
        output_axis = axes[0]
        x_left, x_right = output_axis.get_xlim()
        if x_position > (x_left + x_right) / 2:
            tooltip.set_position((0.62, 0.98))
            tooltip.set_ha("left")
        else:
            tooltip.set_position((0.02, 0.98))
            tooltip.set_ha("left")
        tooltip.set_text(
            "\n".join(
                [
                    record.name,
                    f"Date: {date_labels[hovered_index]}",
                    f"Output: ${record.output_price_per_million:.4g}/M",
                    f"Input: ${record.input_price_per_million:.4g}/M",
                ]
            )
        )
        tooltip.set_visible(True)
        figure.canvas.draw_idle()

    def on_leave(_event) -> None:
        hide_guides()

    figure.canvas.mpl_connect("motion_notify_event", on_move)
    figure.canvas.mpl_connect("axes_leave_event", on_leave)


def build_plot(records: list[ModelRecord]) -> Figure:
    x_positions = list(range(len(records)))
    date_labels = [record.released_at.strftime("%Y-%m-%d") for record in records]
    output_prices = [record.output_price_per_million for record in records]
    input_prices = [record.input_price_per_million for record in records]

    figure, (output_axis, input_axis) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1]},
    )

    output_axis.plot(
        x_positions,
        output_prices,
        marker="s",
        linewidth=1.4,
        markersize=4.5,
        color="#d62728",
        alpha=0.85,
        label="Output price ($/M tokens)",
    )
    output_axis.set_title("OpenRouter VLM Models In Sequence")
    output_axis.set_ylabel("Output price ($/M)")
    output_axis.grid(True, which="major", linestyle=":", alpha=0.35)
    output_axis.grid(True, which="minor", linestyle=":", alpha=0.15)
    output_axis.legend(loc="upper left")

    input_axis.plot(
        x_positions,
        input_prices,
        marker="o",
        linewidth=1.4,
        markersize=4.5,
        color="#9467bd",
        alpha=0.9,
        label="Input price ($/M tokens)",
    )
    input_axis.set_ylabel("Input price ($/M)")
    input_axis.set_xlabel("Model sequence")
    input_axis.grid(True, which="major", linestyle=":", alpha=0.35)
    input_axis.legend(loc="upper left")

    tick_step = max(1, len(x_positions) // 20)
    tick_positions = x_positions[::tick_step]
    tick_labels = date_labels[::tick_step]
    if x_positions and tick_positions[-1] != x_positions[-1]:
        tick_positions.append(x_positions[-1])
        tick_labels.append(date_labels[-1])
    input_axis.set_xticks(tick_positions)
    input_axis.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    figure.tight_layout()
    attach_interaction(
        figure,
        [output_axis, input_axis],
        records,
        x_positions,
        date_labels,
        output_prices,
        input_prices,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display OpenRouter VLM pricing plots over time."
    )
    parser.add_argument(
        "--input",
        default="openrouter_VLM.txt",
        help="Path to the OpenRouter model summary text file.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the generated figure.",
    )
    parser.add_argument(
        "--max-output-price",
        type=float,
        default=DEFAULT_MAX_OUTPUT_PRICE_PER_MILLION,
        help="Keep only models with output price less than or equal to this value. Use a negative value to disable filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    max_output_price = args.max_output_price
    if max_output_price is not None and max_output_price < 0:
        max_output_price = None
    records = parse_records(input_path, max_output_price)
    figure = build_plot(records)
    if args.save:
        output_path = Path(args.save).resolve()
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot with {len(records)} records to {output_path}")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
