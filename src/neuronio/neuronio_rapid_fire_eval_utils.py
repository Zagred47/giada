from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RapidFireConfig:
    ignore_time_at_start_ms: int = 500
    smooth_window: int = 7
    high_quantile: float = 0.85
    rapid_quantile: float = 0.85
    merge_gap: int = 10
    min_event_length: int = 2
    peak_search_before: int = 8
    peak_search_after: int = 45
    window_before: int = 40
    window_after: int = 80
    bad_peak_error_thresholds: List[float] = field(
        default_factory=lambda: [0.8, 1.0, 1.5, 2.0, 3.0]
    )


@dataclass
class RapidFireEvent:
    sim_index: int
    onset_time: int
    end_time: int
    peak_time: int
    true_peak_value: float
    max_derivative: float
    mean_derivative: float


@dataclass
class RapidFireEvaluationResult:
    metrics: Dict[str, float]
    events: List[RapidFireEvent]
    event_metrics: Dict[str, np.ndarray] = field(default_factory=dict)


class RapidFireEventDetector:
    def __init__(self, config: Optional[RapidFireConfig] = None):
        self.config = config if config is not None else RapidFireConfig()
        self.high_threshold = np.nan
        self.rapid_derivative_threshold = np.nan

    def detect(self, y_soma_gt: np.ndarray) -> List[RapidFireEvent]:
        y_soma_gt = self._as_2d_array(y_soma_gt)
        events = []

        high_threshold = self.compute_high_threshold(y_soma_gt)
        rapid_threshold = self.compute_rapid_derivative_threshold(
            y_soma_gt, high_threshold
        )

        self.high_threshold = high_threshold
        self.rapid_derivative_threshold = rapid_threshold

        if np.isnan(high_threshold) or np.isnan(rapid_threshold):
            return events

        for sim_index in range(y_soma_gt.shape[0]):
            y = y_soma_gt[sim_index]
            y_smooth = self.smooth(y)
            dy = np.gradient(y_smooth)

            rapid_mask = (y >= high_threshold) & (dy >= rapid_threshold)
            rapid_mask[: self.config.ignore_time_at_start_ms] = False

            for onset_time, end_time in self.segment_mask(rapid_mask):
                search_start = max(0, onset_time - self.config.peak_search_before)
                search_end = min(len(y), end_time + self.config.peak_search_after)
                peak_time = self.local_max_time(y, search_start, search_end)

                if peak_time is None:
                    continue

                event_dy = dy[onset_time : end_time + 1]
                events.append(
                    RapidFireEvent(
                        sim_index=int(sim_index),
                        onset_time=int(onset_time),
                        end_time=int(end_time),
                        peak_time=int(peak_time),
                        true_peak_value=float(y[peak_time]),
                        max_derivative=float(event_dy.max()),
                        mean_derivative=float(event_dy.mean()),
                    )
                )

        return events

    def compute_high_threshold(self, y_soma_gt: np.ndarray) -> float:
        y_soma_gt = self._as_2d_array(y_soma_gt)
        values = y_soma_gt[:, self.config.ignore_time_at_start_ms :].reshape(-1)
        if len(values) == 0:
            return np.nan
        return float(np.quantile(values, self.config.high_quantile))

    def compute_rapid_derivative_threshold(
        self, y_soma_gt: np.ndarray, high_threshold: float
    ) -> float:
        y_soma_gt = self._as_2d_array(y_soma_gt)
        derivative_values = []

        for sim_index in range(y_soma_gt.shape[0]):
            y = y_soma_gt[sim_index]
            y_smooth = self.smooth(y)
            dy = np.gradient(y_smooth)

            valid_mask = (y >= high_threshold) & (dy > 0)
            valid_mask[: self.config.ignore_time_at_start_ms] = False
            derivative_values.extend(dy[valid_mask].tolist())

        if len(derivative_values) == 0:
            return np.nan

        return float(np.quantile(derivative_values, self.config.rapid_quantile))

    def smooth(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.config.smooth_window <= 1:
            return x.copy()

        kernel = np.ones(self.config.smooth_window) / self.config.smooth_window
        return np.convolve(x, kernel, mode="same")

    def segment_mask(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        indices = np.where(mask)[0]
        segments = []

        if len(indices) == 0:
            return segments

        start = int(indices[0])
        last = int(indices[0])

        for t in indices[1:]:
            t = int(t)
            if t - last <= self.config.merge_gap:
                last = t
            else:
                if last - start + 1 >= self.config.min_event_length:
                    segments.append((start, last))
                start = t
                last = t

        if last - start + 1 >= self.config.min_event_length:
            segments.append((start, last))

        return segments

    def local_max_time(
        self, y: np.ndarray, start: int, stop: int
    ) -> Optional[int]:
        y = np.asarray(y)
        start = max(0, int(start))
        stop = min(len(y), int(stop))

        if stop <= start:
            return None

        return int(start + np.argmax(y[start:stop]))

    def _as_2d_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("Expected a 1D or 2D soma array.")
        return values


class RapidFireSomaEvaluator:
    def __init__(self, config: Optional[RapidFireConfig] = None):
        self.config = config if config is not None else RapidFireConfig()
        self.detector = RapidFireEventDetector(self.config)

    def evaluate(
        self, y_soma_gt: np.ndarray, y_soma_hat: np.ndarray
    ) -> RapidFireEvaluationResult:
        y_soma_gt, y_soma_hat = self._validate_arrays(y_soma_gt, y_soma_hat)
        events = self.detector.detect(y_soma_gt)
        event_metrics = self._compute_event_metrics(y_soma_gt, y_soma_hat, events)

        metrics = {
            "rapid_event_count": float(len(events)),
            "rapid_high_threshold": float(self.detector.high_threshold),
            "rapid_derivative_threshold": float(
                self.detector.rapid_derivative_threshold
            ),
            "soma_corr_global": self._compute_correlation(
                y_soma_gt.reshape(-1), y_soma_hat.reshape(-1)
            ),
            "soma_corr_rapid": self._compute_correlation_on_event_windows(
                y_soma_gt, y_soma_hat, events
            ),
            "soma_rmse_rapid": self._compute_rmse_on_event_windows(
                y_soma_gt, y_soma_hat, events
            ),
        }
        metrics.update(self._compute_coverage_metrics(y_soma_gt, events))
        metrics.update(self._compute_peak_error_summary(event_metrics))

        return RapidFireEvaluationResult(
            metrics=metrics, events=events, event_metrics=event_metrics
        )

    def print_results(self, result: RapidFireEvaluationResult) -> None:
        metrics = result.metrics

        print("")
        print("Rapid-fire soma diagnostics")
        print("--------------------------------")
        print(f"Rapid events          : {int(metrics['rapid_event_count'])}")
        print(f"High threshold        : {metrics['rapid_high_threshold']:.4f}")
        print(f"Derivative threshold  : {metrics['rapid_derivative_threshold']:.4f}")
        print(f"Soma corr global      : {metrics['soma_corr_global']:.4f}")
        print(f"Soma corr rapid       : {metrics['soma_corr_rapid']:.4f}")
        print(f"Soma RMSE rapid       : {metrics['soma_rmse_rapid']:.4f}")
        print(f"Peak abs error mean   : {metrics['peak_abs_error_mean']:.4f}")
        print(f"Peak signed error mean: {metrics['peak_signed_error_mean']:.4f}")

        print("")
        print("Rapid-fire coverage")
        print("--------------------------------")
        print(f"Total eval points     : {int(metrics['total_eval_time_points'])}")
        print(f"Rapid segment points  : {int(metrics['rapid_segment_points'])}")
        print(f"Rapid segment percent : {metrics['rapid_segment_percent']:.2f}%")
        print(f"Rapid window points   : {int(metrics['rapid_window_points'])}")
        print(f"Rapid window percent  : {metrics['rapid_window_percent']:.2f}%")
        print(f"Total simulations     : {int(metrics['total_simulations'])}")
        print(f"Simulations with rapid: {int(metrics['simulations_with_rapid'])}")
        print(f"Simulation percent    : {metrics['simulations_with_rapid_percent']:.2f}%")

        print("")
        print("Rapid-fire peak error distribution")
        print("--------------------------------")
        print(f"Peak abs error median : {metrics['peak_abs_error_median']:.4f}")
        print(f"Peak abs error p90    : {metrics['peak_abs_error_p90']:.4f}")
        print(f"Peak abs error p95    : {metrics['peak_abs_error_p95']:.4f}")
        print(f"Peak abs error p99    : {metrics['peak_abs_error_p99']:.4f}")
        print(f"Underestimated peaks  : {int(metrics['underestimated_peak_count'])}")
        print(f"Underestimated percent: {metrics['underestimated_peak_percent']:.2f}%")

        for threshold in self.config.bad_peak_error_thresholds:
            key_suffix = self._threshold_key_suffix(threshold)
            count_key = f"bad_peak_error_count_at_{key_suffix}"
            percent_key = f"bad_peak_error_percent_at_{key_suffix}"
            print(
                f"Abs peak error >= {threshold:g}: "
                f"{int(metrics[count_key])} ({metrics[percent_key]:.2f}%)"
            )

    def plot_event(
        self,
        y_soma_gt: np.ndarray,
        y_soma_hat: np.ndarray,
        result: RapidFireEvaluationResult,
        event_index: int = 0,
        save_fig_path: Optional[str] = None,
    ) -> None:
        y_soma_gt, y_soma_hat = self._validate_arrays(y_soma_gt, y_soma_hat)
        if event_index < 0 or event_index >= len(result.events):
            raise IndexError("event_index is outside the available rapid-fire events.")

        event = result.events[event_index]
        self._plot_single_event(y_soma_gt, y_soma_hat, event, save_fig_path)

    def plot_worst_peak_errors(
        self,
        y_soma_gt: np.ndarray,
        y_soma_hat: np.ndarray,
        result: RapidFireEvaluationResult,
        n: int = 4,
        save_dir: Optional[str] = None,
    ) -> None:
        y_soma_gt, y_soma_hat = self._validate_arrays(y_soma_gt, y_soma_hat)
        scored_events = []

        for event in result.events:
            true_peak = y_soma_gt[event.sim_index, event.peak_time]
            pred_peak = y_soma_hat[event.sim_index, event.peak_time]
            scored_events.append((abs(pred_peak - true_peak), event))

        scored_events = sorted(scored_events, key=lambda row: row[0], reverse=True)

        for idx, (_, event) in enumerate(scored_events[:n]):
            save_fig_path = None
            if save_dir is not None:
                save_fig_path = (
                    f"{save_dir}/rapid_fire_worst_peak_error_{idx:02d}.png"
                )
            self._plot_single_event(y_soma_gt, y_soma_hat, event, save_fig_path)

    def _plot_single_event(
        self,
        y_soma_gt: np.ndarray,
        y_soma_hat: np.ndarray,
        event: RapidFireEvent,
        save_fig_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        y_true_full = y_soma_gt[event.sim_index]
        y_pred_full = y_soma_hat[event.sim_index]

        start = max(0, event.onset_time - self.config.window_before)
        stop = min(len(y_true_full), event.peak_time + self.config.window_after)
        t = np.arange(start, stop)

        y_smooth = self.detector.smooth(y_true_full)
        dy = np.gradient(y_smooth)
        error = y_pred_full - y_true_full

        color_palette = sns.color_palette("colorblind")
        sns.set_palette(color_palette)
        sns.set_style("white")

        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        title = (
            f"Rapid-fire event | sim={event.sim_index} "
            f"onset={event.onset_time} peak={event.peak_time}"
        )

        axs[0].plot(t, y_true_full[start:stop], label="Target soma", linewidth=1.2)
        axs[0].plot(
            t, y_pred_full[start:stop], label="Predicted soma", linewidth=1.2
        )
        axs[0].axvline(event.onset_time, linestyle="--", alpha=0.5)
        axs[0].axvline(event.peak_time, linestyle=":", alpha=0.8)
        axs[0].set_ylabel("Soma")
        axs[0].set_title(title)
        axs[0].legend()

        axs[1].plot(t, error[start:stop], label="Prediction error", linewidth=1.2)
        axs[1].axhline(0, linestyle="--", alpha=0.5)
        axs[1].axvline(event.onset_time, linestyle="--", alpha=0.5)
        axs[1].axvline(event.peak_time, linestyle=":", alpha=0.8)
        axs[1].set_ylabel("Error")
        axs[1].legend()

        axs[2].plot(t, dy[start:stop], label="Target soma derivative", linewidth=1.2)
        axs[2].axhline(
            self.detector.rapid_derivative_threshold,
            linestyle="--",
            alpha=0.7,
            label="Rapid threshold",
        )
        axs[2].axvline(event.onset_time, linestyle="--", alpha=0.5)
        axs[2].axvline(event.peak_time, linestyle=":", alpha=0.8)
        axs[2].set_ylabel("Derivative")
        axs[2].set_xlabel("Time (ms)")
        axs[2].legend()

        fig.tight_layout()
        if save_fig_path is not None:
            fig.savefig(save_fig_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

    def _compute_correlation_on_event_windows(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        events: List[RapidFireEvent],
    ) -> float:
        true_values, pred_values = self._collect_event_window_values(
            y_true, y_pred, events
        )
        return self._compute_correlation(true_values, pred_values)

    def _compute_rmse_on_event_windows(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        events: List[RapidFireEvent],
    ) -> float:
        true_values, pred_values = self._collect_event_window_values(
            y_true, y_pred, events
        )
        return self._compute_rmse(true_values, pred_values)

    def _collect_event_window_values(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        events: List[RapidFireEvent],
    ) -> Tuple[np.ndarray, np.ndarray]:
        true_values = []
        pred_values = []

        for event in events:
            start = max(0, event.onset_time - self.config.window_before)
            stop = min(y_true.shape[1], event.peak_time + self.config.window_after)

            true_values.extend(y_true[event.sim_index, start:stop].tolist())
            pred_values.extend(y_pred[event.sim_index, start:stop].tolist())

        return np.asarray(true_values), np.asarray(pred_values)

    def _compute_event_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        events: List[RapidFireEvent],
    ) -> Dict[str, np.ndarray]:
        peak_abs_errors = []
        peak_signed_errors = []
        event_durations = []
        window_durations = []

        for event in events:
            true_peak = y_true[event.sim_index, event.peak_time]
            pred_peak = y_pred[event.sim_index, event.peak_time]
            signed_error = pred_peak - true_peak

            start = max(
                self.config.ignore_time_at_start_ms,
                event.onset_time - self.config.window_before,
            )
            stop = min(y_true.shape[1], event.peak_time + self.config.window_after)

            peak_abs_errors.append(abs(signed_error))
            peak_signed_errors.append(signed_error)
            event_durations.append(event.end_time - event.onset_time + 1)
            window_durations.append(max(0, stop - start))

        return {
            "peak_abs_error": np.asarray(peak_abs_errors, dtype=float),
            "peak_signed_error": np.asarray(peak_signed_errors, dtype=float),
            "event_duration": np.asarray(event_durations, dtype=float),
            "window_duration": np.asarray(window_durations, dtype=float),
        }

    def _compute_coverage_metrics(
        self, y_true: np.ndarray, events: List[RapidFireEvent]
    ) -> Dict[str, float]:
        num_sims, num_time = y_true.shape
        ignore = self.config.ignore_time_at_start_ms
        total_eval_time_points = num_sims * max(0, num_time - ignore)

        rapid_segment_mask = np.zeros_like(y_true, dtype=bool)
        rapid_window_mask = np.zeros_like(y_true, dtype=bool)
        sims_with_rapid = set()

        for event in events:
            sims_with_rapid.add(event.sim_index)

            segment_start = max(ignore, event.onset_time)
            segment_stop = min(num_time, event.end_time + 1)
            if segment_stop > segment_start:
                rapid_segment_mask[event.sim_index, segment_start:segment_stop] = True

            window_start = max(ignore, event.onset_time - self.config.window_before)
            window_stop = min(num_time, event.peak_time + self.config.window_after)
            if window_stop > window_start:
                rapid_window_mask[event.sim_index, window_start:window_stop] = True

        rapid_segment_points = int(rapid_segment_mask.sum())
        rapid_window_points = int(rapid_window_mask.sum())
        simulations_with_rapid = len(sims_with_rapid)

        if total_eval_time_points == 0:
            rapid_segment_fraction = np.nan
            rapid_window_fraction = np.nan
        else:
            rapid_segment_fraction = rapid_segment_points / total_eval_time_points
            rapid_window_fraction = rapid_window_points / total_eval_time_points

        if num_sims == 0:
            simulation_fraction = np.nan
        else:
            simulation_fraction = simulations_with_rapid / num_sims

        return {
            "total_eval_time_points": float(total_eval_time_points),
            "rapid_segment_points": float(rapid_segment_points),
            "rapid_segment_fraction": float(rapid_segment_fraction),
            "rapid_segment_percent": float(100 * rapid_segment_fraction),
            "rapid_window_points": float(rapid_window_points),
            "rapid_window_fraction": float(rapid_window_fraction),
            "rapid_window_percent": float(100 * rapid_window_fraction),
            "total_simulations": float(num_sims),
            "simulations_with_rapid": float(simulations_with_rapid),
            "simulations_with_rapid_fraction": float(simulation_fraction),
            "simulations_with_rapid_percent": float(100 * simulation_fraction),
        }

    def _compute_peak_error_summary(
        self, event_metrics: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        abs_errors = event_metrics.get("peak_abs_error", np.asarray([]))
        signed_errors = event_metrics.get("peak_signed_error", np.asarray([]))
        summary = {}

        if len(abs_errors) == 0:
            summary.update(
                {
                    "peak_abs_error_mean": np.nan,
                    "peak_abs_error_median": np.nan,
                    "peak_abs_error_p90": np.nan,
                    "peak_abs_error_p95": np.nan,
                    "peak_abs_error_p99": np.nan,
                    "peak_signed_error_mean": np.nan,
                    "underestimated_peak_count": 0.0,
                    "underestimated_peak_fraction": np.nan,
                    "underestimated_peak_percent": np.nan,
                }
            )
            for threshold in self.config.bad_peak_error_thresholds:
                key_suffix = self._threshold_key_suffix(threshold)
                summary[f"bad_peak_error_count_at_{key_suffix}"] = 0.0
                summary[f"bad_peak_error_fraction_at_{key_suffix}"] = np.nan
                summary[f"bad_peak_error_percent_at_{key_suffix}"] = np.nan
            return summary

        underestimated = signed_errors < 0
        summary.update(
            {
                "peak_abs_error_mean": float(np.mean(abs_errors)),
                "peak_abs_error_median": float(np.median(abs_errors)),
                "peak_abs_error_p90": float(np.quantile(abs_errors, 0.90)),
                "peak_abs_error_p95": float(np.quantile(abs_errors, 0.95)),
                "peak_abs_error_p99": float(np.quantile(abs_errors, 0.99)),
                "peak_signed_error_mean": float(np.mean(signed_errors)),
                "underestimated_peak_count": float(underestimated.sum()),
                "underestimated_peak_fraction": float(underestimated.mean()),
                "underestimated_peak_percent": float(100 * underestimated.mean()),
            }
        )

        for threshold in self.config.bad_peak_error_thresholds:
            bad = abs_errors >= threshold
            key_suffix = self._threshold_key_suffix(threshold)
            summary[f"bad_peak_error_count_at_{key_suffix}"] = float(bad.sum())
            summary[f"bad_peak_error_fraction_at_{key_suffix}"] = float(bad.mean())
            summary[f"bad_peak_error_percent_at_{key_suffix}"] = float(100 * bad.mean())

        return summary

    def _compute_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)

        if len(x) < 3 or len(y) < 3:
            return np.nan
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return np.nan

        return float(np.corrcoef(x, y)[0, 1])

    def _compute_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0:
            return np.nan

        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def _threshold_key_suffix(self, threshold: float) -> str:
        return str(float(threshold)).replace(".", "_")

    def _validate_arrays(
        self, y_soma_gt: np.ndarray, y_soma_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_soma_gt = np.asarray(y_soma_gt, dtype=float)
        y_soma_hat = np.asarray(y_soma_hat, dtype=float)

        if y_soma_gt.ndim == 1:
            y_soma_gt = y_soma_gt[None, :]
        if y_soma_hat.ndim == 1:
            y_soma_hat = y_soma_hat[None, :]

        if y_soma_gt.shape != y_soma_hat.shape:
            raise ValueError("y_soma_gt and y_soma_hat must have the same shape.")
        if y_soma_gt.ndim != 2:
            raise ValueError("Expected 1D or 2D soma arrays.")

        return y_soma_gt, y_soma_hat
