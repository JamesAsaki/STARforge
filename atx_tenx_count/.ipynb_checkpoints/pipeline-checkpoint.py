from __future__ import annotations

import gzip
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


sns.set_context("paper")
sns.set_style("whitegrid")

STAGE_LABELS = {
    "01_parsed": "Stage 01 - Parse and Aggregate FASTQs",
    "02_mapped": "Stage 02 - Map Barcodes",
    "03_umi_collapsed": "Stage 03 - Collapse UMIs",
    "04_collision_resolved": "Stage 04 - Resolve Cross-Feature Collisions",
}

STAGE_REPORT_ENTITIES = {
    "parsed": [
        ("cell_bc", "raw cell barcode"),
        ("umi", "molecule barcode"),
        ("hekumi", "technical barcode"),
        ("mhc_bc", "raw MHC barcode"),
        ("pep_bc", "raw peptide barcode"),
    ],
    "mapped": [
        ("cell_id", "mapped cell identifier"),
        ("umi", "molecule barcode"),
        ("hekumi", "technical barcode"),
        ("mhc_id", "mapped MHC identifier"),
        ("pep_id", "mapped peptide identifier"),
        ("feature_id", "composite feature identifier"),
    ],
}

R2_PRE26 = "GTGCCGTCCGTGTCCATTCACTCGAG"
BASES = "ACGT"

_PARSED_SCHEMA: dict[str, pl.DataType] = {
    "cell_bc": pl.String,
    "umi": pl.String,
    "mhc_bc": pl.String,
    "pep_bc": pl.String,
    "hekumi": pl.String,
    "reads": pl.Int64,
    "cell_q": pl.Int64,
    "umi_q": pl.Int64,
    "mhc_q": pl.Int64,
    "pep_q": pl.Int64,
    "hekumi_q": pl.Int64,
}

_LOOKUP_SCHEMA = {
    "bc": pl.String,
    "id": pl.String,
    "mm": pl.Int64,
    "status": pl.String,
    "match": pl.String,
}

_UMI_MAP_SCHEMA: dict[str, pl.DataType] = {
    "cell_id": pl.String,
    "mhc_id": pl.String,
    "pep_id": pl.String,
    "umi": pl.String,
    "umi_collapsed": pl.String,
}

_STAGE03_EVENT_SCHEMA: dict[str, pl.DataType] = {
    "cell_id": pl.String,
    "mhc_id": pl.String,
    "pep_id": pl.String,
    "feature_id": pl.String,
    "pass": pl.Int64,
    "src_umi": pl.String,
    "tgt_umi": pl.String,
    "src_reads": pl.Int64,
    "tgt_reads": pl.Int64,
    "ratio": pl.Float64,
}

_STAGE03_GROUP_SCHEMA: dict[str, pl.DataType] = {
    "cell_id": pl.String,
    "mhc_id": pl.String,
    "pep_id": pl.String,
    "feature_id": pl.String,
    "group_reads": pl.Int64,
    "n_collapses": pl.Int64,
    "n_umi_before": pl.Int64,
    "n_umi_after": pl.Int64,
}

_STAGE03_SCHEMA: dict[str, pl.DataType] = {
    "cell_id": pl.String,
    "umi": pl.String,
    "mhc_bc": pl.String,
    "pep_bc": pl.String,
    "hekumi": pl.String,
    "mhc_id": pl.String,
    "pep_id": pl.String,
    "feature_id": pl.String,
    "mhc_mm": pl.Int64,
    "pep_mm": pl.Int64,
    "reads": pl.Int64,
}

_STAGE04_COLLISION_SCHEMA: dict[str, pl.DataType] = {
    "cell_id": pl.String,
    "umi": pl.String,
    "top_feature": pl.String,
    "second_feature": pl.String,
    "top_reads": pl.Int64,
    "second_reads": pl.Int64,
    "ratio": pl.Float64,
    "action": pl.String,
}

_STAGE04_DIRECTED_RATIO_SCHEMA: dict[str, pl.DataType] = {
    "winner_feature": pl.String,
    "loser_feature": pl.String,
    "ratio": pl.Float64,
}

_STAGE04_SCHEMA = _STAGE03_SCHEMA.copy()

_Q_PROB = [0.0] * 128
for q in range(94):
    _Q_PROB[q + 33] = 1.0 - 10 ** (-q / 10.0)


@dataclass(slots=True)
class PipelineConfig:
    sample_name: str
    r1_fastq: Path
    r2_fastq: Path
    cell_barcodes: Path
    out_dir: Path
    fb_ref: Path | None = None
    mhc_ref: Path | None = None
    pep_ref: Path | None = None
    max_reads_parse: int | None = None
    save_intermediates: bool = True
    generate_graphs: bool = True
    plot_dpi: int = 60
    require_strict_constants: bool = True
    umi_collapse_max_passes: int = 3
    umi_min_ratio: float = 5.0
    collision_min_ratio: float = 5.0
    collision_min_top_reads: int = 3
    min_avg_q: float = 99.05

    def __post_init__(self) -> None:
        self.r1_fastq = Path(self.r1_fastq)
        self.r2_fastq = Path(self.r2_fastq)
        self.cell_barcodes = Path(self.cell_barcodes)
        self.out_dir = Path(self.out_dir)
        self.fb_ref = Path(self.fb_ref) if self.fb_ref is not None else None
        self.mhc_ref = Path(self.mhc_ref) if self.mhc_ref is not None else None
        self.pep_ref = Path(self.pep_ref) if self.pep_ref is not None else None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.generate_graphs:
            self.plot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def plot_dir(self) -> Path:
        return self.out_dir / "plots"

    @property
    def parquet_01(self) -> Path:
        return self.out_dir / f"{self.sample_name}_01_parsed.parquet"

    @property
    def parquet_02(self) -> Path:
        return self.out_dir / f"{self.sample_name}_02_mapped.parquet"

    @property
    def parquet_03(self) -> Path:
        return self.out_dir / f"{self.sample_name}_03_umi_collapsed.parquet"

    @property
    def parquet_03_group_metrics(self) -> Path:
        return self.out_dir / f"{self.sample_name}_03_umi_group_metrics.parquet"

    @property
    def parquet_03_collapse_events(self) -> Path:
        return self.out_dir / f"{self.sample_name}_03_umi_collapse_events.parquet"

    @property
    def parquet_04(self) -> Path:
        return self.out_dir / f"{self.sample_name}_04_global_collision_resolved.parquet"

    @property
    def parquet_04_collision_metrics(self) -> Path:
        return self.out_dir / f"{self.sample_name}_04_collision_metrics.parquet"

    @property
    def parquet_04_directed_ratio_events(self) -> Path:
        return self.out_dir / f"{self.sample_name}_04_directed_ratio_events.parquet"


def empty_dataframe(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 10**9
    return sum(x != y for x, y in zip(a, b))


def parse_fb(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    out = (
        df.select(
            pl.col("feature_bc").alias("bc"),
            pl.col("feature_id").alias("fid"),
        )
        .with_columns(
            pl.col("bc").str.len_chars().alias("len"),
            pl.col("bc").str.slice(0, 15).alias("mhc_bc"),
            pl.col("bc").str.slice(15, 4).alias("mid"),
            pl.col("bc").str.slice(19, 25).alias("pep_bc"),
            pl.col("fid").str.split_exact("|", 1).alias("parts"),
        )
        .with_columns(
            pl.col("parts").struct.field("field_0").alias("mhc_id"),
            pl.col("parts").struct.field("field_1").alias("pep_id"),
        )
    )

    bad = out.filter(
        (pl.col("len") != 44)
        | (pl.col("mid") != "TTCC")
        | pl.col("mhc_id").is_null()
        | pl.col("pep_id").is_null()
    )
    if bad.height:
        raise ValueError("Invalid feature_bc or feature_id format.")

    return (
        out.select(["mhc_id", "mhc_bc"]).unique(),
        out.select(["pep_id", "pep_bc"]).unique(),
    )


def convert_q_scores(qual: str) -> int:
    n = len(qual)
    if n == 0:
        return 0
    s = 0.0
    for ch in qual:
        s += _Q_PROB[ord(ch)]
    return int(s * 100.0 / n + 0.5)


def _fastq_read_id(header: str) -> str:
    h = header.strip()
    token = h.split()[0]
    if token.endswith("/1") or token.endswith("/2"):
        token = token[:-2]
    return token


def parse_fastq_and_aggregate(
    r1_path: Path,
    r2_path: Path,
    require_constants: bool = True,
    max_reads: int | None = None,
    track_quality: bool = True,
    min_avg_q: float = 99.05,
) -> tuple[pl.DataFrame, Counter]:
    reads_counter: Counter[tuple[str, str, str, str, str]] = Counter()
    qc: Counter[str] = Counter()
    qsum: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])

    with gzip.open(r1_path, "rt") as r1, gzip.open(r2_path, "rt") as r2:
        while True:
            h1 = r1.readline()
            h2 = r2.readline()
            if not h1 and not h2:
                break
            if (not h1) != (not h2):
                raise ValueError("R1 and R2 ended at different times")

            s1 = r1.readline().rstrip("\n")
            s2 = r2.readline().rstrip("\n")
            r1.readline()
            r2.readline()
            q1 = r1.readline().rstrip("\n")
            q2 = r2.readline().rstrip("\n")

            qc["records_seen"] += 1
            if max_reads is not None and qc["records_seen"] > max_reads:
                break

            if qc["records_seen"] % 100_000 == 0:
                id1 = _fastq_read_id(h1)
                id2 = _fastq_read_id(h2)
                if id1 != id2:
                    raise ValueError(f"R1/R2 header mismatch: {id1} != {id2}")
                print(f"\rProgress: {qc['records_seen']} reads", end="", flush=True)

            if len(s1) != 28 or len(s2) != 90:
                qc["filtered_length"] += 1
                continue

            qc["pass_length"] += 1
            pre_ok = s2[:26] == R2_PRE26
            tt_ok = s2[41:45] == "TTCC"
            ac_ok = s2[70:74] == "ACTC"
            if require_constants and not (pre_ok and tt_ok and ac_ok):
                qc["filtered_constants"] += 1
                continue

            qc["pass_constants"] += 1

            key = (
                s1[:16],
                s1[16:28],
                s2[26:41],
                s2[45:70],
                s2[74:90],
            )
            reads_counter[key] += 1

            cell_q = convert_q_scores(q1[:16])
            umi_q = convert_q_scores(q1[16:28])
            mhc_q = convert_q_scores(q2[26:41])
            pep_q = convert_q_scores(q2[45:70])
            hek_q = convert_q_scores(q2[74:90])

            if track_quality:
                sums = qsum[key]
                sums[0] += cell_q
                sums[1] += umi_q
                sums[2] += mhc_q
                sums[3] += pep_q
                sums[4] += hek_q

    if qc["records_seen"] >= 100_000:
        print()

    rows: list[dict[str, Any]] = []
    low_q = 0
    for key, reads in reads_counter.items():
        sums = qsum[key]
        cell_q = sums[0] / reads if track_quality else 100.0
        umi_q = sums[1] / reads if track_quality else 100.0
        mhc_q = sums[2] / reads if track_quality else 100.0
        pep_q = sums[3] / reads if track_quality else 100.0
        hek_q = sums[4] / reads if track_quality else 100.0

        if umi_q > min_avg_q and cell_q > min_avg_q:
            rows.append(
                {
                    "cell_bc": key[0],
                    "umi": key[1],
                    "mhc_bc": key[2],
                    "pep_bc": key[3],
                    "hekumi": key[4],
                    "reads": reads,
                    "cell_q": int(round(cell_q)),
                    "umi_q": int(round(umi_q)),
                    "mhc_q": int(round(mhc_q)),
                    "pep_q": int(round(pep_q)),
                    "hekumi_q": int(round(hek_q)),
                }
            )
        else:
            low_q += 1

    qc["low_q_filtered"] = low_q
    if not rows:
        return empty_dataframe(_PARSED_SCHEMA), qc
    return pl.DataFrame(rows), qc


def make_spans(block_lengths: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for n in block_lengths:
        end = start + n
        spans.append((start, end))
        start = end
    return spans


def build_seed_index(sequences: list[str], block_lengths: list[int]) -> tuple[dict[tuple[int, str], set[str]], list[tuple[int, int]]]:
    if not sequences:
        return {}, []

    seq_len = len(sequences[0])
    if sum(block_lengths) != seq_len:
        raise ValueError("block_lengths must sum to sequence length")

    spans = make_spans(block_lengths)
    index: dict[tuple[int, str], set[str]] = defaultdict(set)
    for seq in sequences:
        if len(seq) != seq_len:
            raise ValueError("all sequences must have same length")
        for i, (start, end) in enumerate(spans):
            block = seq[start:end]
            index[(i, block)].add(seq)

    return index, spans


def map_one_sequence(
    query: str,
    whitelist_set: set[str],
    seq_to_id: dict[str, str],
    seed_index: dict[tuple[int, str], set[str]],
    spans: list[tuple[int, int]],
    max_mm: int,
) -> tuple[str, int, str, str]:
    if query in whitelist_set:
        return seq_to_id[query], 0, "exact", query

    candidates: set[str] = set()
    for i, (start, end) in enumerate(spans):
        block = query[start:end]
        candidates.update(seed_index.get((i, block), ()))

    best_mm = max_mm + 1
    best: list[str] = []
    for cand in candidates:
        d = hamming_distance(query, cand)
        if d < best_mm:
            best_mm = d
            best = [cand]
        elif d == best_mm:
            best.append(cand)

    if best_mm <= max_mm:
        if len(best) == 1:
            target = best[0]
            return seq_to_id[target], best_mm, "mismatch", target
        return "unknown", best_mm, "ambiguous", ""

    return "unknown", -1, "unknown", ""


def map_unique_barcodes(
    queries: list[str],
    whitelist_seq_to_id: dict[str, str],
    max_mm: int,
    block_lengths: list[int],
    label: str,
) -> pl.DataFrame:
    whitelist = list(whitelist_seq_to_id.keys())
    whitelist_set = set(whitelist)
    seed_index, spans = build_seed_index(whitelist, block_lengths)

    records = []
    status_ct: Counter[str] = Counter()
    mm_ct: Counter[int] = Counter()
    n_total = len(queries)

    for i, q in enumerate(queries, start=1):
        mapped_id, mm, status, matched_seq = map_one_sequence(
            q,
            whitelist_set,
            whitelist_seq_to_id,
            seed_index,
            spans,
            max_mm,
        )
        status_ct[status] += 1
        if mm >= 0:
            mm_ct[mm] += 1
        records.append(
            {
                label: q,
                f"{label}_id": mapped_id,
                f"{label}_mm": mm,
                f"{label}_status": status,
                f"{label}_match": matched_seq,
            }
        )

        if i % 1_000 == 0:
            print(f"\r{label}: mapped {i:,}/{n_total:,}", end="", flush=True)

    if n_total >= 1_000:
        print()
    print(f"{label} status counts (unique query):", dict(status_ct))
    print(f"{label} mismatch counts (unique query):", dict(mm_ct))

    if not records:
        return pl.DataFrame(
            {
                label: pl.Series([], dtype=pl.String),
                f"{label}_id": pl.Series([], dtype=pl.String),
                f"{label}_mm": pl.Series([], dtype=pl.Int64),
                f"{label}_status": pl.Series([], dtype=pl.String),
                f"{label}_match": pl.Series([], dtype=pl.String),
            }
        )
    return pl.DataFrame(records)


def hamming1_neighbors(seq: str):
    for i, ch in enumerate(seq):
        for b in BASES:
            if b != ch:
                yield seq[:i] + b + seq[i + 1 :]


def collapse_umis_iterative(umi_counts: dict[str, int], min_ratio: float = 3.0, max_passes: int = 3):
    counts = dict(umi_counts)
    mapping = {u: u for u in counts}
    events = []

    for p in range(1, max_passes + 1):
        changed = False
        for src in sorted(list(counts), key=lambda u: (counts[u], u)):
            if src not in counts:
                continue

            src_reads = counts[src]
            best_tgt = None
            best_tgt_reads = -1

            for tgt in hamming1_neighbors(src):
                tgt_reads = counts.get(tgt)
                if tgt_reads is None or tgt_reads <= src_reads:
                    continue
                if (tgt_reads / src_reads) < min_ratio:
                    continue
                if tgt_reads > best_tgt_reads or (
                    tgt_reads == best_tgt_reads and (best_tgt is None or tgt < best_tgt)
                ):
                    best_tgt = tgt
                    best_tgt_reads = tgt_reads

            if best_tgt is None:
                continue

            events.append(
                {
                    "pass": p,
                    "src_umi": src,
                    "tgt_umi": best_tgt,
                    "src_reads": src_reads,
                    "tgt_reads": counts[best_tgt],
                    "ratio": counts[best_tgt] / src_reads,
                }
            )

            counts[best_tgt] += src_reads
            del counts[src]
            for orig, curr in mapping.items():
                if curr == src:
                    mapping[orig] = best_tgt
            changed = True

        if not changed:
            break

    return mapping, events, counts


def counter_to_df(
    counter: dict[str, int] | Counter,
    key_name: str,
    value_name: str = "count",
    sort_desc: bool | None = True,
) -> pl.DataFrame:
    rows = [{key_name: key, value_name: int(value)} for key, value in counter.items()]
    if not rows:
        return pl.DataFrame({key_name: [], value_name: []})

    df = pl.DataFrame(rows)
    if sort_desc is True:
        return df.sort([value_name, key_name], descending=[True, False])
    if sort_desc is False:
        return df.sort([value_name, key_name], descending=[False, False])
    return df


def metrics_mapping_df(
    metrics: dict[str, object],
    key_name: str = "metric",
    value_name: str = "value",
) -> pl.DataFrame:
    return pl.DataFrame([{key_name: key, value_name: str(value)} for key, value in metrics.items()])


def stage_plot_path(plot_root: Path, stage_key: str, sample_name: str, plot_slug: str) -> Path:
    stage_dir = Path(plot_root) / stage_key
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / f"{sample_name}_{stage_key}_{plot_slug}.png"


def stage_report(
    df: pl.DataFrame,
    stage_key: str,
    config: PipelineConfig,
    extra_tables: list[tuple[str, pl.DataFrame | dict[str, object]]] | None = None,
) -> dict[str, Any]:
    cols = set(df.columns)
    has_reads = "reads" in cols
    rows = df.height
    total_reads = int(df["reads"].sum()) if has_reads and rows else (0 if has_reads else None)

    schema_key = "mapped" if ({"cell_id", "feature_id"} & cols) else "parsed"
    entity_specs = STAGE_REPORT_ENTITIES[schema_key]

    summary_rows = []
    for entity, note in entity_specs:
        if entity not in cols:
            summary_rows.append(
                {"entity": entity, "unique_count": None, "total_reads": None, "notes": f"{note} (missing)"}
            )
            continue

        entity_reads = None
        if has_reads:
            entity_reads = int(df.filter(pl.col(entity).is_not_null())["reads"].sum() or 0)

        summary_rows.append(
            {
                "entity": entity,
                "unique_count": int(df[entity].n_unique()),
                "total_reads": entity_reads,
                "notes": note,
            }
        )

    summary_df = pl.DataFrame(summary_rows)

    print(STAGE_LABELS.get(stage_key, stage_key))
    print("rows", rows)
    print("total_reads", total_reads if total_reads is not None else "NA")
    print(summary_df)

    normalized_extra_tables: list[tuple[str, pl.DataFrame]] = []
    for title, table in extra_tables or []:
        if isinstance(table, dict):
            table_df = metrics_mapping_df(table)
        else:
            table_df = table
        print(title)
        print(table_df if table_df.height else "no rows")
        normalized_extra_tables.append((title, table_df))

    plot_paths: dict[str, Path | None] = {}
    if not config.generate_graphs:
        return {
            "stage_key": stage_key,
            "rows": rows,
            "total_reads": total_reads,
            "summary_df": summary_df,
            "extra_tables": normalized_extra_tables,
            "plot_paths": plot_paths,
        }

    def _finalize_figure(plot_slug: str) -> None:
        plot_path = stage_plot_path(config.plot_dir, stage_key, config.sample_name, plot_slug)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print("Saved", plot_path)
        plot_paths[plot_slug] = plot_path

    def _histplot_values(
        values: np.ndarray,
        plot_slug: str,
        title: str,
        xlabel: str,
        ylabel: str,
        *,
        log_x: bool = False,
        log_y: bool = False,
        use_log_bins: bool = False,
        n_bins: int = 50,
    ) -> None:
        arr = np.asarray(values)
        arr = arr[arr > 0]
        if arr.size == 0:
            plot_paths[plot_slug] = None
            return

        if use_log_bins and arr.min() != arr.max():
            bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), n_bins)
        else:
            bins = n_bins

        plt.figure(figsize=(8, 4), dpi=config.plot_dpi)
        sns.histplot(arr, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
        _finalize_figure(plot_slug)

    def _group_metric_counts(group_col: str, agg_expr: pl.Expr, value_col: str) -> np.ndarray | None:
        if not has_reads or group_col not in cols:
            return None
        grouped = df.group_by(group_col).agg(agg_expr)
        return grouped[value_col].to_numpy()

    pep_col = "pep_id" if schema_key == "mapped" else "pep_bc"
    mhc_col = "mhc_id" if schema_key == "mapped" else "mhc_bc"
    cell_col = "cell_id" if schema_key == "mapped" else "cell_bc"

    pep_values = _group_metric_counts(pep_col, pl.sum("reads").alias("reads"), "reads")
    if pep_values is not None:
        _histplot_values(
            pep_values,
            "hist_reads_by_peptide",
            "Reads per peptide",
            "Reads",
            "Count of peptide entries",
            log_x=True,
            log_y=True,
            use_log_bins=True,
        )

    mhc_values = _group_metric_counts(mhc_col, pl.sum("reads").alias("reads"), "reads")
    if mhc_values is not None:
        _histplot_values(
            mhc_values,
            "hist_reads_by_mhc",
            "Reads per MHC",
            "Reads",
            "Count of MHC entries",
        )

    umi_values = _group_metric_counts("umi", pl.sum("reads").alias("reads"), "reads")
    if umi_values is not None:
        _histplot_values(
            umi_values,
            "hist_reads_by_umi",
            "Reads per UMI",
            "Reads",
            "Count of UMIs",
            log_x=True,
            log_y=True,
            use_log_bins=True,
        )

    if has_reads and {"umi", "umi_q"}.issubset(cols):
        umi_reads = df.group_by("umi").agg(pl.sum("reads").alias("reads"), pl.mean("umi_q").alias("q"))
        plt.figure(figsize=(8, 8), dpi=config.plot_dpi)
        sns.scatterplot(data=umi_reads.to_pandas(), x="reads", y="q")
        plt.title("Reads per UMI by quality")
        plt.xlabel("Reads")
        plt.ylabel("avg quality")
        plt.xscale("log")
        _finalize_figure("reads_by_quality")

    hek_per_umi_values = _group_metric_counts("hekumi", pl.n_unique("umi").alias("n_umi"), "n_umi")
    if hek_per_umi_values is not None:
        _histplot_values(
            hek_per_umi_values,
            "hist_umi_per_hekumi",
            "Unique UMI per HEKUMI",
            "# unique umi per hekumi",
            "Count of hekumi",
            log_x=True,
            log_y=True,
            use_log_bins=True,
        )

    umi_per_cell_values = _group_metric_counts(cell_col, pl.n_unique("umi").alias("n_umi"), "n_umi")
    if umi_per_cell_values is not None:
        _histplot_values(
            umi_per_cell_values,
            "hist_umi_per_cell",
            "Unique UMI per cell",
            "# unique umi per cell",
            "Count of cells",
            log_x=True,
            log_y=True,
            use_log_bins=True,
        )

    reads_per_cell_values = _group_metric_counts(cell_col, pl.sum("reads").alias("sum_reads"), "sum_reads")
    if reads_per_cell_values is not None:
        _histplot_values(
            reads_per_cell_values,
            "hist_reads_per_cell",
            "Reads per cell",
            "# reads per cell",
            "Count of cells",
            log_x=True,
            log_y=True,
            use_log_bins=True,
        )

    hekumi_per_cell_values = _group_metric_counts(cell_col, pl.n_unique("hekumi").alias("n_hekumi"), "n_hekumi")
    if hekumi_per_cell_values is not None:
        _histplot_values(
            hekumi_per_cell_values,
            "hist_hekumi_per_cell",
            "Unique HEKUMI per cell",
            "# unique hekumi per cell",
            "Count of cells",
            log_x=True,
            log_y=True,
            use_log_bins=True,
        )

    return {
        "stage_key": stage_key,
        "rows": rows,
        "total_reads": total_reads,
        "summary_df": summary_df,
        "extra_tables": normalized_extra_tables,
        "plot_paths": plot_paths,
    }


def write_if_enabled(df: pl.DataFrame, path: Path, enabled: bool) -> None:
    if not enabled:
        return
    df.write_parquet(path)
    print("Wrote", path)


def load_references(config: PipelineConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    if config.fb_ref:
        if config.mhc_ref or config.pep_ref:
            raise ValueError("Invalid config: fb_ref cannot be provided together with mhc_ref or pep_ref.")
    else:
        if config.mhc_ref and config.pep_ref:
            pass
        elif config.mhc_ref or config.pep_ref:
            raise ValueError("Invalid config: Both mhc_ref and pep_ref must be provided together.")
        else:
            raise ValueError("Invalid config: Provide either fb_ref alone OR both mhc_ref and pep_ref.")

    if config.fb_ref:
        fb_ref_df = pl.read_csv(config.fb_ref)
        mhc_ref_df, pep_ref_df = parse_fb(fb_ref_df)
        reference_summary_df = metrics_mapping_df(
            {
                "feature_id": fb_ref_df["feature_id"].n_unique(),
                "mhc_id": mhc_ref_df["mhc_id"].n_unique(),
                "pep_id": pep_ref_df["pep_id"].n_unique(),
            }
        )
    else:
        fb_ref_df = None
        mhc_ref_df = pl.read_csv(config.mhc_ref)
        pep_ref_df = pl.read_csv(config.pep_ref)
        reference_summary_df = metrics_mapping_df(
            {
                "mhc_id": mhc_ref_df["mhc_id"].n_unique(),
                "pep_id": pep_ref_df["pep_id"].n_unique(),
            }
        )

    print("Reference summary")
    print(reference_summary_df)
    return mhc_ref_df, pep_ref_df, fb_ref_df


def stage01_parse(config: PipelineConfig) -> tuple[pl.DataFrame, Counter, dict[str, Any]]:
    parsed_df, parse_qc = parse_fastq_and_aggregate(
        config.r1_fastq,
        config.r2_fastq,
        require_constants=config.require_strict_constants,
        max_reads=config.max_reads_parse,
        track_quality=True,
        min_avg_q=config.min_avg_q,
    )

    rows = parsed_df.height
    reads = int(parsed_df["reads"].sum()) if rows else 0
    estimated_saturation = ((reads - rows) / reads) if reads else None
    dedup_keys = ["cell_bc", "umi", "mhc_bc", "pep_bc", "hekumi"]

    parse_qc_df = counter_to_df(
        {
            key: int(parse_qc.get(key, 0))
            for key in [
                "records_seen",
                "pass_length",
                "filtered_length",
                "pass_constants",
                "filtered_constants",
                "low_q_filtered",
            ]
        },
        key_name="metric",
        value_name="count",
        sort_desc=None,
    )
    parse_details_df = metrics_mapping_df(
        {
            "parsed_aggregated_rows": f"{rows:,}",
            "total_reads_after_parse_filters": f"{reads:,}",
            "estimated_unfiltered_sequencing_saturation": (
                f"{estimated_saturation:.2%}" if estimated_saturation is not None else "NA"
            ),
            "dedup_keys": ", ".join(dedup_keys),
        }
    )

    write_if_enabled(parsed_df, config.parquet_01, config.save_intermediates)
    report = stage_report(
        parsed_df,
        "01_parsed",
        config,
        extra_tables=[("Parse QC", parse_qc_df), ("Parse Details", parse_details_df)],
    )
    return parsed_df, parse_qc, report


def stage02_map(
    config: PipelineConfig,
    parsed_df: pl.DataFrame,
    mhc_ref_df: pl.DataFrame,
    pep_ref_df: pl.DataFrame,
    fb_ref_df: pl.DataFrame | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    cellbc_ref = pl.read_csv(
        config.cell_barcodes,
        separator="\t",
        has_header=False,
        new_columns=["bc_ref"],
    ).with_columns(pl.col("bc_ref").str.replace(r"-1$", ""))

    pep_seq_to_id = dict(zip(pep_ref_df["pep_bc"].to_list(), pep_ref_df["pep_id"].to_list()))
    mhc_seq_to_id = dict(zip(mhc_ref_df["mhc_bc"].to_list(), mhc_ref_df["mhc_id"].to_list()))
    cellbarcode_map = dict(zip(cellbc_ref["bc_ref"].to_list(), cellbc_ref["bc_ref"].to_list()))
    valid_features = set(fb_ref_df["feature_id"].to_list()) if fb_ref_df is not None else None

    unique_pep_queries = parsed_df["pep_bc"].unique().to_list()
    unique_mhc_queries = parsed_df["mhc_bc"].unique().to_list()
    unique_cellbc_queries = parsed_df["cell_bc"].unique().to_list()

    print("Begin pep mapping")
    pep_lookup_df = map_unique_barcodes(
        unique_pep_queries,
        pep_seq_to_id,
        max_mm=4,
        block_lengths=[5, 5, 5, 5, 5],
        label="pep_bc",
    ).rename(
        {
            "pep_bc_id": "pep_id",
            "pep_bc_mm": "pep_mm",
            "pep_bc_status": "pep_status",
            "pep_bc_match": "pep_match",
        }
    )

    print("Begin mhc mapping")
    mhc_lookup_df = map_unique_barcodes(
        unique_mhc_queries,
        mhc_seq_to_id,
        max_mm=2,
        block_lengths=[5, 5, 5],
        label="mhc_bc",
    ).rename(
        {
            "mhc_bc_id": "mhc_id",
            "mhc_bc_mm": "mhc_mm",
            "mhc_bc_status": "mhc_status",
            "mhc_bc_match": "mhc_match",
        }
    )

    print("Begin cellbc mapping")
    cellbc_lookup_df = map_unique_barcodes(
        unique_cellbc_queries,
        cellbarcode_map,
        max_mm=1,
        block_lengths=[8, 8],
        label="cell_bc",
    ).rename(
        {
            "cell_bc_id": "cell_id",
            "cell_bc_mm": "cell_mm",
            "cell_bc_status": "cell_status",
            "cell_bc_match": "cell_match",
        }
    )

    df_map = (
        parsed_df.join(pep_lookup_df, on="pep_bc", how="left")
        .join(mhc_lookup_df, on="mhc_bc", how="left")
        .join(cellbc_lookup_df, on="cell_bc", how="left")
    )

    pep_unique_status_df = (
        pep_lookup_df.group_by("pep_status").agg(pl.len().alias("unique_queries")).sort("unique_queries", descending=True)
    )
    mhc_unique_status_df = (
        mhc_lookup_df.group_by("mhc_status").agg(pl.len().alias("unique_queries")).sort("unique_queries", descending=True)
    )
    cell_unique_status_df = (
        cellbc_lookup_df.group_by("cell_status")
        .agg(pl.len().alias("unique_queries"))
        .sort("unique_queries", descending=True)
    )

    pep_weighted = df_map.group_by("pep_status").agg(pl.sum("reads").alias("reads")).sort("reads", descending=True)
    mhc_weighted = df_map.group_by("mhc_status").agg(pl.sum("reads").alias("reads")).sort("reads", descending=True)
    cell_weighted = df_map.group_by("cell_status").agg(pl.sum("reads").alias("reads")).sort("reads", descending=True)

    pep_mm_weighted = (
        df_map.filter(pl.col("pep_status").is_in(["exact", "mismatch"]))
        .group_by("pep_mm")
        .agg(pl.sum("reads").alias("reads"))
        .sort("pep_mm")
    )
    mhc_mm_weighted = (
        df_map.filter(pl.col("mhc_status").is_in(["exact", "mismatch"]))
        .group_by("mhc_mm")
        .agg(pl.sum("reads").alias("reads"))
        .sort("mhc_mm")
    )
    cell_mm_weighted = (
        df_map.filter(pl.col("cell_status").is_in(["exact", "mismatch"]))
        .group_by("cell_mm")
        .agg(pl.sum("reads").alias("reads"))
        .sort("cell_mm")
    )

    pep_amb_reads = int(df_map.filter(pl.col("pep_status") == "ambiguous")["reads"].sum() or 0)
    mhc_amb_reads = int(df_map.filter(pl.col("mhc_status") == "ambiguous")["reads"].sum() or 0)
    cell_amb_reads = int(df_map.filter(pl.col("cell_status") == "ambiguous")["reads"].sum() or 0)

    df_mapped = df_map.filter(
        pl.col("pep_status").is_in(["exact", "mismatch"])
        & pl.col("mhc_status").is_in(["exact", "mismatch"])
        & pl.col("cell_status").is_in(["exact", "mismatch", "unknown"])
    )

    df_mapped = df_mapped.with_columns(
        pl.when(pl.col("cell_status") == "unknown")
        .then(pl.col("cell_bc"))
        .otherwise(pl.col("cell_id"))
        .alias("cell_id")
    )

    df_mapped = df_mapped.with_columns((pl.col("mhc_id") + pl.lit("|") + pl.col("pep_id")).alias("feature_id"))

    before_feature_filter = df_mapped.height
    if valid_features is not None:
        df_mapped = df_mapped.filter(pl.col("feature_id").is_in(valid_features))
    rows_dropped_invalid_features = before_feature_filter - df_mapped.height

    mapping_query_counts_df = pl.DataFrame(
        [
            {"entity": "pep_bc", "unique_queries": len(unique_pep_queries)},
            {"entity": "mhc_bc", "unique_queries": len(unique_mhc_queries)},
            {"entity": "cell_bc", "unique_queries": len(unique_cellbc_queries)},
        ]
    )
    ambiguous_reads_df = pl.DataFrame(
        [
            {"entity": "pep_bc", "ambiguous_reads": pep_amb_reads},
            {"entity": "mhc_bc", "ambiguous_reads": mhc_amb_reads},
            {"entity": "cell_bc", "ambiguous_reads": cell_amb_reads},
        ]
    )
    feature_filter_df = metrics_mapping_df(
        {"rows_dropped_for_invalid_feature_combinations": rows_dropped_invalid_features}
    )

    write_if_enabled(df_mapped, config.parquet_02, config.save_intermediates)
    report = stage_report(
        df_mapped,
        "02_mapped",
        config,
        extra_tables=[
            ("Mapping Query Counts", mapping_query_counts_df),
            ("Peptide Unique Query Status", pep_unique_status_df),
            ("Peptide Weighted Status", pep_weighted),
            ("Peptide Mismatch Buckets", pep_mm_weighted),
            ("MHC Unique Query Status", mhc_unique_status_df),
            ("MHC Weighted Status", mhc_weighted),
            ("MHC Mismatch Buckets", mhc_mm_weighted),
            ("Cell Barcode Unique Query Status", cell_unique_status_df),
            ("Cell Barcode Weighted Status", cell_weighted),
            ("Cell Barcode Mismatch Buckets", cell_mm_weighted),
            ("Ambiguous Reads", ambiguous_reads_df),
            ("Feature Filter", feature_filter_df),
        ],
    )
    return df_mapped, report


def stage03_collapse(config: PipelineConfig, df_mapped: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    group_umi_df_m2 = (
        df_mapped.group_by(["cell_id", "mhc_id", "pep_id", "feature_id", "umi"]).agg(pl.sum("reads").alias("umi_reads"))
    )

    groups_m2: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    group_feature_m2: dict[tuple[str, str, str], str] = {}
    for row in group_umi_df_m2.iter_rows(named=True):
        key = (row["cell_id"], row["mhc_id"], row["pep_id"])
        groups_m2[key][row["umi"]] = int(row["umi_reads"])
        group_feature_m2[key] = row["feature_id"]

    umi_map_rows_m2 = []
    collapse_event_rows_m2 = []
    group_metric_rows_m2 = []

    for (cell_id, mhc_id, pep_id), umi_counts in groups_m2.items():
        feature_id = group_feature_m2[(cell_id, mhc_id, pep_id)]
        mapping, events, final_counts = collapse_umis_iterative(
            umi_counts,
            min_ratio=config.umi_min_ratio,
            max_passes=config.umi_collapse_max_passes,
        )

        umi_map_rows_m2.extend(
            {
                "cell_id": cell_id,
                "mhc_id": mhc_id,
                "pep_id": pep_id,
                "umi": orig,
                "umi_collapsed": corr,
            }
            for orig, corr in mapping.items()
        )
        collapse_event_rows_m2.extend(
            {"cell_id": cell_id, "mhc_id": mhc_id, "pep_id": pep_id, "feature_id": feature_id, **event}
            for event in events
        )
        group_metric_rows_m2.append(
            {
                "cell_id": cell_id,
                "mhc_id": mhc_id,
                "pep_id": pep_id,
                "feature_id": feature_id,
                "group_reads": int(sum(umi_counts.values())),
                "n_collapses": len(events),
                "n_umi_before": len(umi_counts),
                "n_umi_after": len(final_counts),
            }
        )

    umi_map_df_m2 = pl.DataFrame(umi_map_rows_m2) if umi_map_rows_m2 else empty_dataframe(_UMI_MAP_SCHEMA)
    collapse_events_df_m2 = (
        pl.DataFrame(collapse_event_rows_m2) if collapse_event_rows_m2 else empty_dataframe(_STAGE03_EVENT_SCHEMA)
    )
    group_metrics_df_m2 = (
        pl.DataFrame(group_metric_rows_m2) if group_metric_rows_m2 else empty_dataframe(_STAGE03_GROUP_SCHEMA)
    )

    df_umi_m2 = (
        df_mapped.join(umi_map_df_m2, on=["cell_id", "mhc_id", "pep_id", "umi"], how="left")
        .with_columns(pl.col("umi_collapsed").fill_null(pl.col("umi")).alias("umi"))
        .drop("umi_collapsed")
        .group_by(
            [
                "cell_id",
                "umi",
                "mhc_bc",
                "pep_bc",
                "hekumi",
                "mhc_id",
                "pep_id",
                "feature_id",
                "mhc_mm",
                "pep_mm",
            ]
        )
        .agg(pl.sum("reads").alias("reads"))
    )
    if df_umi_m2.height == 0:
        df_umi_m2 = empty_dataframe(_STAGE03_SCHEMA)

    write_if_enabled(group_metrics_df_m2, config.parquet_03_group_metrics, config.save_intermediates)
    write_if_enabled(collapse_events_df_m2, config.parquet_03_collapse_events, config.save_intermediates)
    write_if_enabled(df_umi_m2, config.parquet_03, config.save_intermediates)

    stage03_summary_df = metrics_mapping_df(
        {
            "umi_collapse_groups": len(groups_m2),
            "umi_collapse_events": collapse_events_df_m2.height,
            "output_rows": df_umi_m2.height,
            "output_reads": int(df_umi_m2["reads"].sum()) if df_umi_m2.height else 0,
        }
    )
    report = stage_report(
        df_umi_m2,
        "03_umi_collapsed",
        config,
        extra_tables=[("UMI Collapse Summary", stage03_summary_df)],
    )
    return df_umi_m2, group_metrics_df_m2, collapse_events_df_m2, report


def stage03_diagnostics(
    config: PipelineConfig,
    df_umi_m2: pl.DataFrame,
    group_metrics_df_m2: pl.DataFrame,
    collapse_events_df_m2: pl.DataFrame,
) -> None:
    print("Stage 03 diagnostics")
    print("df_umi_m2 shape=", df_umi_m2.shape)
    print("group_metrics_df_m2 shape=", group_metrics_df_m2.shape)
    print("collapse_events_df_m2 shape=", collapse_events_df_m2.shape)

    feature_summary = df_umi_m2.group_by("feature_id").agg(pl.n_unique("umi").alias("n_umi")).sort("n_umi", descending=True)
    top20_pl = feature_summary.head(20)
    bottom20_pl = feature_summary.tail(20).sort("n_umi")
    print("Top 20 features by UMI")
    print(top20_pl)
    print("Bottom 20 features by UMI")
    print(bottom20_pl)

    if not config.generate_graphs:
        return

    stage03_plot_dir = config.plot_dir / "03_umi_collapsed"
    stage03_plot_dir.mkdir(parents=True, exist_ok=True)

    scatter_df = pd.DataFrame(group_metrics_df_m2.select(["group_reads", "n_collapses"]).to_dicts())
    if not scatter_df.empty:
        plt.figure(figsize=(7, 5), dpi=config.plot_dpi)
        sns.scatterplot(data=scatter_df, x="group_reads", y="n_collapses", s=12)
        plt.xscale("log")
        plt.title("UMI collapse: reads vs collapses")
        plt.xlabel("Group reads")
        plt.ylabel("UMI collapses")
        fn = stage03_plot_dir / f"{config.sample_name}_03_umi_collapsed_umi_collapse_scatter.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        print("Saved", fn)

    if collapse_events_df_m2.height > 0:
        plt.figure(figsize=(8, 4), dpi=config.plot_dpi)
        sns.histplot(collapse_events_df_m2["ratio"].to_list(), bins=100)
        plt.title("UMI collapse ratio (target/source)")
        plt.xlabel("Ratio")
        plt.ylabel("Collapse events")
        fn = stage03_plot_dir / f"{config.sample_name}_03_umi_collapsed_umi_collapse_ratio_hist.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        print("Saved", fn)
    else:
        print("No UMI collapse events")

    top20 = pd.DataFrame(top20_pl.to_dicts())
    bottom20 = pd.DataFrame(bottom20_pl.to_dicts())
    if not top20.empty or not bottom20.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=config.plot_dpi)
        if not top20.empty:
            sns.barplot(data=top20, y="feature_id", x="n_umi", ax=axes[0], color="#4c78a8")
        axes[0].set_title("Top 20 features by unique UMI")
        axes[0].set_xlabel("Unique UMI")
        axes[0].set_ylabel("")
        if not bottom20.empty:
            sns.barplot(data=bottom20, y="feature_id", x="n_umi", ax=axes[1], color="#f58518")
        axes[1].set_title("Bottom 20 features by unique UMI")
        axes[1].set_xlabel("Unique UMI")
        axes[1].set_ylabel("")
        fn = stage03_plot_dir / f"{config.sample_name}_03_umi_collapsed_feature_umi_top_bottom20.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        print("Saved", fn)


def stage04_observed_collision(config: PipelineConfig, df_umi_m2: pl.DataFrame) -> pl.DataFrame:
    df_pair = df_umi_m2.with_columns((pl.col("cell_id") + pl.lit("|") + pl.col("umi")).alias("cell_umi"))

    feature_pair_m2 = (
        df_pair.group_by("feature_id")
        .agg(pl.col("cell_umi").unique().alias("pair_list"), pl.n_unique("cell_umi").alias("n_pair"))
        .sort("n_pair", descending=True)
    )
    features_pair = feature_pair_m2["feature_id"].to_list()
    pair_sets = {row["feature_id"]: set(row["pair_list"]) for row in feature_pair_m2.iter_rows(named=True)}

    obs_pair_mat = np.zeros((len(features_pair), len(features_pair)), dtype=np.float32)
    for i, fa in enumerate(features_pair):
        set_a = pair_sets[fa]
        denom = max(len(set_a), 1)
        for j, fb in enumerate(features_pair):
            inter = len(set_a.intersection(pair_sets[fb]))
            obs_pair_mat[i, j] = 100.0 * inter / denom

    if config.generate_graphs and len(features_pair) > 0:
        stage04_plot_dir = config.plot_dir / "04_collision_resolved"
        stage04_plot_dir.mkdir(parents=True, exist_ok=True)
        norm_obs = LogNorm(vmin=0.001, vmax=100)
        plt.figure(figsize=(9, 7), dpi=config.plot_dpi * 4)
        ax = sns.heatmap(
            np.where(obs_pair_mat <= 0, 1e-6, obs_pair_mat),
            cmap="rocket",
            norm=norm_obs,
            xticklabels=False,
            yticklabels=False,
        )
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.001, 0.01, 0.1, 1, 10, 100])
        cbar.set_ticklabels(["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"])
        plt.title("Observed (cell_id, umi) collision % (A by B)")
        fn = stage04_plot_dir / f"{config.sample_name}_04_collision_resolved_observed_cell_umi_collision_heatmap.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        print("Saved", fn)
        print("Observed (cell_id, umi) heatmap dimensions:", obs_pair_mat.shape)

    pair_stats = df_pair.group_by("cell_umi").agg(
        pl.n_unique("feature_id").alias("n_feat"),
        pl.sum("reads").alias("pair_reads"),
    )
    n_pairs_total = pair_stats.height
    n_pairs_collided = pair_stats.filter(pl.col("n_feat") > 1).height
    reads_total = int(pair_stats["pair_reads"].sum()) if pair_stats.height else 0
    reads_collided = int(pair_stats.filter(pl.col("n_feat") > 1)["pair_reads"].sum() or 0)

    observed_collision_summary_df = metrics_mapping_df(
        {
            "feature_count": len(features_pair),
            "total_cell_id_umi": f"{n_pairs_total:,}",
            "collided_cell_id_umi": f"{n_pairs_collided:,}",
            "pct_pairs_collided": f"{(100.0 * n_pairs_collided / max(n_pairs_total, 1)):.4f}%",
            "total_reads": f"{reads_total:,}",
            "reads_in_collided_pairs": f"{reads_collided:,}",
            "pct_reads_collided": f"{(100.0 * reads_collided / max(reads_total, 1)):.4f}%",
        }
    )
    print("Observed collision summary")
    print(observed_collision_summary_df)
    return observed_collision_summary_df


def stage04_resolve(
    config: PipelineConfig,
    df_umi_m2: pl.DataFrame,
    mhc_ref_df: pl.DataFrame,
    pep_ref_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    feature_parts_df_m2 = (
        df_umi_m2.select("feature_id")
        .unique()
        .with_columns(pl.col("feature_id").str.split_exact("|", 1).alias("parts"))
        .with_columns(
            pl.col("parts").struct.field("field_0").alias("mhc_id"),
            pl.col("parts").struct.field("field_1").alias("pep_id"),
        )
        .drop("parts")
        .join(mhc_ref_df.select(["mhc_id", "mhc_bc"]).unique(), on="mhc_id", how="left")
        .join(pep_ref_df.select(["pep_id", "pep_bc"]).unique(), on="pep_id", how="left")
    )
    feature_parts_m2 = {
        row["feature_id"]: (row["mhc_id"], row["pep_id"], row["mhc_bc"], row["pep_bc"])
        for row in feature_parts_df_m2.iter_rows(named=True)
    }

    feature_reads_df_m2 = (
        df_umi_m2.group_by(["cell_id", "umi", "feature_id"]).agg(pl.sum("reads").alias("feature_reads"))
    )

    collision_groups_m2: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for row in feature_reads_df_m2.iter_rows(named=True):
        collision_groups_m2[(row["cell_id"], row["umi"])].append((row["feature_id"], int(row["feature_reads"])))

    decision_m2: dict[tuple[str, str], dict[str, str]] = {}
    directed_ratio_m2: dict[tuple[str, str], list[float]] = defaultdict(list)
    collision_metric_rows_m2 = []

    for key, flist in collision_groups_m2.items():
        if len(flist) < 2:
            continue

        ranked = sorted(flist, key=lambda x: (-x[1], x[0]))
        (top_f, top_r), (second_f, second_r) = ranked[:2]
        ratio = float("inf") if second_r == 0 else top_r / second_r

        if top_r == second_r:
            action, keep_f = "drop_all_tie", ""
        elif top_r >= config.collision_min_top_reads and ratio > config.collision_min_ratio:
            action, keep_f = "keep_top", top_f
            for loser_f, loser_r in ranked[1:]:
                if loser_r > 0:
                    directed_ratio_m2[(top_f, loser_f)].append(top_r / loser_r)
        else:
            action, keep_f = "drop_all_ratio_or_low_top", ""

        decision_m2[key] = {"action": action, "top_feature": keep_f}
        collision_metric_rows_m2.append(
            {
                "cell_id": key[0],
                "umi": key[1],
                "top_feature": top_f,
                "second_feature": second_f,
                "top_reads": top_r,
                "second_reads": second_r,
                "ratio": ratio,
                "action": action,
            }
        )

    out_counter_m2: Counter[tuple[str, str, str, str, str, str, str, str, int, int]] = Counter()
    reads_reassigned_m2 = 0
    reads_dropped_m2 = 0

    for row in df_umi_m2.iter_rows(named=True):
        key = (row["cell_id"], row["umi"])
        dec = decision_m2.get(key)
        if dec is not None and dec["action"] != "keep_top":
            reads_dropped_m2 += int(row["reads"])
            continue

        keep_feature = row["feature_id"] if dec is None else dec["top_feature"]
        keep_mhc_id, keep_pep_id, keep_mhc_bc, keep_pep_bc = feature_parts_m2[keep_feature]

        if dec is not None and row["feature_id"] != keep_feature:
            reads_reassigned_m2 += int(row["reads"])

        out_key = (
            row["cell_id"],
            row["umi"],
            keep_mhc_bc,
            keep_pep_bc,
            row["hekumi"],
            keep_mhc_id,
            keep_pep_id,
            keep_feature,
            row["mhc_mm"],
            row["pep_mm"],
        )
        out_counter_m2[out_key] += int(row["reads"])

    df_final_m2 = (
        pl.DataFrame(
            [
                {
                    "cell_id": key[0],
                    "umi": key[1],
                    "mhc_bc": key[2],
                    "pep_bc": key[3],
                    "hekumi": key[4],
                    "mhc_id": key[5],
                    "pep_id": key[6],
                    "feature_id": key[7],
                    "mhc_mm": key[8],
                    "pep_mm": key[9],
                    "reads": value,
                }
                for key, value in out_counter_m2.items()
            ]
        )
        if out_counter_m2
        else empty_dataframe(_STAGE04_SCHEMA)
    )

    collision_metrics_df_m2 = (
        pl.DataFrame(collision_metric_rows_m2)
        if collision_metric_rows_m2
        else empty_dataframe(_STAGE04_COLLISION_SCHEMA)
    )
    directed_ratio_df_m2 = (
        pl.DataFrame(
            [
                {"winner_feature": winner, "loser_feature": loser, "ratio": ratio}
                for (winner, loser), vals in directed_ratio_m2.items()
                for ratio in vals
            ]
        )
        if directed_ratio_m2
        else empty_dataframe(_STAGE04_DIRECTED_RATIO_SCHEMA)
    )

    collision_metrics_df_m2.write_parquet(config.parquet_04_collision_metrics)
    print("Wrote", config.parquet_04_collision_metrics)
    directed_ratio_df_m2.write_parquet(config.parquet_04_directed_ratio_events)
    print("Wrote", config.parquet_04_directed_ratio_events)
    df_final_m2.write_parquet(config.parquet_04)
    print("Wrote", config.parquet_04)

    stage04_summary_df = metrics_mapping_df(
        {
            "collision_groups": len(decision_m2),
            "keep_top": sum(d["action"] == "keep_top" for d in decision_m2.values()),
            "drop_all_tie": sum(d["action"] == "drop_all_tie" for d in decision_m2.values()),
            "drop_all_ratio_or_low_top": sum(
                d["action"] == "drop_all_ratio_or_low_top" for d in decision_m2.values()
            ),
            "reads_reassigned": reads_reassigned_m2,
            "reads_dropped": reads_dropped_m2,
            "output_rows": df_final_m2.height,
            "output_reads": int(df_final_m2["reads"].sum()) if df_final_m2.height else 0,
        }
    )
    report = stage_report(
        df_final_m2,
        "04_collision_resolved",
        config,
        extra_tables=[("Collision Resolution Summary", stage04_summary_df)],
    )
    return df_final_m2, collision_metrics_df_m2, directed_ratio_df_m2, report


def stage04_diagnostics(
    config: PipelineConfig,
    df_final_m2: pl.DataFrame,
    collision_metrics_df_m2: pl.DataFrame,
    directed_ratio_df_m2: pl.DataFrame,
) -> None:
    print("Stage 04 diagnostics")
    print("df_final_m2 shape=", df_final_m2.shape)
    print("collision_metrics_df_m2 shape=", collision_metrics_df_m2.shape)
    print("directed_ratio_df_m2 shape=", directed_ratio_df_m2.shape)

    if directed_ratio_df_m2.height > 0:
        directed_count_df = (
            directed_ratio_df_m2.group_by(["winner_feature", "loser_feature"])
            .agg(pl.len().alias("n_umi_collapsed"))
            .sort("n_umi_collapsed", descending=True)
        )
        print("Top winner/loser feature pairs by collapsed UMI count:")
        print(directed_count_df.head(20))
    else:
        directed_count_df = empty_dataframe(
            {"winner_feature": pl.String, "loser_feature": pl.String, "n_umi_collapsed": pl.Int64}
        )
        print("No directed ratio rows; skipping collapsed-UMI top pairs")

    if not config.generate_graphs:
        return

    stage04_plot_dir = config.plot_dir / "04_collision_resolved"
    stage04_plot_dir.mkdir(parents=True, exist_ok=True)

    if collision_metrics_df_m2.height > 0:
        plt.figure(figsize=(8, 4), dpi=config.plot_dpi)
        sns.histplot(collision_metrics_df_m2["ratio"].to_list(), bins=100)
        plt.title("Collision ratio histogram (top / second)")
        plt.xlabel("Top / second ratio")
        plt.ylabel("Collision groups")
        fn = stage04_plot_dir / f"{config.sample_name}_04_collision_resolved_collision_ratio_hist.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        print("Saved", fn)
    else:
        print("No collision metrics rows; skipping ratio histogram")

    pair_reads_final = df_final_m2.group_by(["cell_id", "umi"]).agg(pl.sum("reads").alias("reads"))
    vals = pair_reads_final["reads"].to_numpy()
    vals = vals[vals > 0]
    if len(vals) > 0:
        bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 80) if vals.min() != vals.max() else 80
        plt.figure(figsize=(8, 4), dpi=config.plot_dpi)
        sns.histplot(vals, bins=bins)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Reads per (cell_id, umi)")
        plt.xlabel("Reads")
        plt.ylabel("Count of (cell_id, umi)")
        fn = stage04_plot_dir / f"{config.sample_name}_04_collision_resolved_reads_per_cellid_umi.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        print("Saved", fn)
    else:
        print("No positive read values for (cell_id, umi) histogram")

    if directed_ratio_df_m2.height > 0:
        feature_order = (
            df_final_m2.group_by("feature_id")
            .agg(pl.struct(["cell_id", "umi"]).n_unique().alias("n_umi"))
            .sort("n_umi", descending=True)
        )["feature_id"].to_list()
        idx = {feature: i for i, feature in enumerate(feature_order)}
        count_mat = np.full((len(feature_order), len(feature_order)), np.nan, dtype=np.float32)
        for row in directed_count_df.iter_rows(named=True):
            winner = row["winner_feature"]
            loser = row["loser_feature"]
            if winner in idx and loser in idx:
                count_mat[idx[winner], idx[loser]] = float(row["n_umi_collapsed"])

        plt.figure(figsize=(9, 7), dpi=config.plot_dpi * 3)
        sns.heatmap(count_mat, cmap="magma", xticklabels=False, yticklabels=False)
        plt.title("Collapsed UMI count heatmap (winner <- loser)")
        fn = stage04_plot_dir / f"{config.sample_name}_04_collision_resolved_directed_umi_count_heatmap.png"
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        print("Saved", fn)
        print("Directed UMI-count heatmap dimensions:", count_mat.shape)


def stage05_summary(
    parsed_df: pl.DataFrame,
    df_mapped: pl.DataFrame,
    df_umi_m2: pl.DataFrame,
    df_final_m2: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    stage_order = ["01_parsed", "02_mapped", "03_umi_collapsed", "04_collision_resolved"]
    stage_dfs = {
        "01_parsed": parsed_df,
        "02_mapped": df_mapped,
        "03_umi_collapsed": df_umi_m2,
        "04_collision_resolved": df_final_m2,
    }

    stage_flow_rows = []
    stage01_reads = None
    previous_reads = None

    for stage_key in stage_order:
        stage_df = stage_dfs[stage_key]
        stage_reads = int(stage_df["reads"].sum()) if "reads" in stage_df.columns else None
        if stage01_reads is None:
            stage01_reads = stage_reads

        retention_vs_previous = None
        if previous_reads not in (None, 0) and stage_reads is not None:
            retention_vs_previous = stage_reads / previous_reads

        retention_vs_stage_01 = None
        if stage01_reads not in (None, 0) and stage_reads is not None:
            retention_vs_stage_01 = stage_reads / stage01_reads

        stage_flow_rows.append(
            {
                "stage": stage_key,
                "rows": stage_df.height,
                "total_reads": stage_reads,
                "retention_vs_previous": f"{retention_vs_previous:.4f}" if retention_vs_previous is not None else "NA",
                "retention_vs_stage_01": f"{retention_vs_stage_01:.4f}" if retention_vs_stage_01 is not None else "NA",
            }
        )
        previous_reads = stage_reads

    stage_flow_df = pl.DataFrame(stage_flow_rows)
    print("Stage flow summary")
    print(stage_flow_df)

    uniqueness_rows = []
    for stage_key in stage_order:
        stage_df = stage_dfs[stage_key]
        entities = (
            ["cell_bc", "umi", "hekumi", "mhc_bc", "pep_bc"]
            if stage_key == "01_parsed"
            else ["cell_id", "umi", "hekumi", "mhc_id", "pep_id", "feature_id"]
        )
        for entity in entities:
            uniqueness_rows.append(
                {
                    "stage": stage_key,
                    "entity": entity,
                    "unique_count": int(stage_df[entity].n_unique()) if entity in stage_df.columns else None,
                }
            )

    uniqueness_df = pl.DataFrame(uniqueness_rows)
    print("Key uniqueness summary")
    print(uniqueness_df)

    reads_01 = stage_flow_rows[0]["total_reads"]
    reads_02 = stage_flow_rows[1]["total_reads"]
    reads_03 = stage_flow_rows[2]["total_reads"]
    reads_04 = stage_flow_rows[3]["total_reads"]

    warnings = []
    if reads_01 is not None and reads_02 is not None and reads_02 > reads_01:
        warnings.append("Stage 02 total_reads exceeds Stage 01 total_reads")
    if reads_02 is not None and reads_03 is not None and reads_03 != reads_02:
        warnings.append("Stage 03 total_reads does not match Stage 02 total_reads")
    if reads_03 is not None and reads_04 is not None and reads_04 > reads_03:
        warnings.append("Stage 04 total_reads exceeds Stage 03 total_reads")

    if warnings:
        print("Warnings")
        for warning in warnings:
            print("-", warning)
    else:
        print("Read-retention invariants passed")

    return stage_flow_df, uniqueness_df


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    print(f"R1: {config.r1_fastq}")
    print(f"R2: {config.r2_fastq}")
    print(f"Output dir: {config.out_dir}")

    mhc_ref_df, pep_ref_df, fb_ref_df = load_references(config)
    parsed_df, parse_qc, stage01_report = stage01_parse(config)
    df_mapped, stage02_report = stage02_map(config, parsed_df, mhc_ref_df, pep_ref_df, fb_ref_df)
    df_umi_m2, group_metrics_df_m2, collapse_events_df_m2, stage03_report = stage03_collapse(config, df_mapped)
    stage03_diagnostics(config, df_umi_m2, group_metrics_df_m2, collapse_events_df_m2)
    observed_collision_summary_df = stage04_observed_collision(config, df_umi_m2)
    df_final_m2, collision_metrics_df_m2, directed_ratio_df_m2, stage04_report = stage04_resolve(
        config,
        df_umi_m2,
        mhc_ref_df,
        pep_ref_df,
    )
    stage04_diagnostics(config, df_final_m2, collision_metrics_df_m2, directed_ratio_df_m2)
    stage_flow_df, uniqueness_df = stage05_summary(parsed_df, df_mapped, df_umi_m2, df_final_m2)

    return {
        "parse_qc": parse_qc,
        "stage01_report": stage01_report,
        "stage02_report": stage02_report,
        "stage03_report": stage03_report,
        "stage04_report": stage04_report,
        "observed_collision_summary_df": observed_collision_summary_df,
        "stage_flow_df": stage_flow_df,
        "uniqueness_df": uniqueness_df,
        "final_df": df_final_m2,
        "collision_metrics_df": collision_metrics_df_m2,
        "directed_ratio_df": directed_ratio_df_m2,
    }
