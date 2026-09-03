"""
Phase 3.2 — CPU/RAM tropical topology on saved SAE + J-lens tensors.

Loads safetensors from disk (no GPU). Computes d_M(x,y) = -ln π(y|x),
Neighbor-Joining trees (Type A), and tropical triangle-inequality checks
(especially Type B contextual shifts).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from .config import (
    DEFAULT_OUTPUT_DIR,
    LAYER_INDEX,
    TRIANGLE_ATOL,
    analysis_summary_json,
    distances_jsonl,
    manifest_jsonl,
    nj_newick_jsonl,
    tensor_path,
    triangle_csv,
    tropical_logprobs_jsonl,
)
from .neighbor_joining import newick_from_distances
from .tropical import (
    reconstruct_sae_vector,
    sae_tropical_distance,
    triangle_slack,
    tropical_distance_from_logprob,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sae_from_file(path: Path) -> torch.Tensor:
    tensors = load_file(str(path))
    d_sae = int(tensors["d_sae"].reshape(-1)[0].item())
    return reconstruct_sae_vector(tensors["sae_indices"], tensors["sae_values"], d_sae)


def _pair_map(trop_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in trop_rows:
        out[str(row["item_id"])][str(row["pair_name"])] = row
    return out


def _sae_waypoint_distances(
    item: dict[str, Any],
    *,
    layer_index: int,
    output_dir: Path,
) -> dict[str, float]:
    iid = str(item["item_id"])
    names = [wp["name"] for wp in item.get("waypoints", [])]
    vecs: dict[str, torch.Tensor] = {}
    for name in names:
        path = tensor_path(iid, name, layer_index, output_dir)
        if path.exists():
            vecs[name] = _sae_from_file(path)
    dists: dict[str, float] = {}
    keys = list(vecs)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            d = sae_tropical_distance(vecs[a], vecs[b])
            dists[f"sae_{a}_{b}"] = d
            dists[f"sae_{b}_{a}"] = d
    return dists


def analyze_item(
    item: dict[str, Any],
    trop_pairs: dict[str, dict[str, Any]],
    *,
    layer_index: int,
    output_dir: Path,
    atol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    iid = str(item["item_id"])
    key = str(item.get("type_key", ""))
    token_d = {
        name: float(p["tropical_distance"])
        for name, p in trop_pairs.items()
        if math_isfinite(p.get("tropical_distance"))
    }
    sae_d = _sae_waypoint_distances(item, layer_index=layer_index, output_dir=output_dir)

    dist_row: dict[str, Any] = {
        "item_id": iid,
        "type_key": key,
        "structure_type": item.get("structure_type"),
        "layer_index": layer_index,
        "token_distances": token_d,
        "sae_distances": sae_d,
    }

    tri_rows: list[dict[str, Any]] = []
    nj_row: dict[str, Any] | None = None

    if key == "A":
        labels = ["n1", "n2", "n3", "n4"]
        # Fill a 4×4 matrix from consecutive + skip token distances;
        # SAE distances fill the remaining undirected pairs.
        dmat = _type_a_distance_matrix(token_d, sae_d)
        newick = newick_from_distances(labels, dmat)
        d12, d23, d34, d14 = (
            token_d.get("n1_n2"),
            token_d.get("n2_n3"),
            token_d.get("n3_n4"),
            token_d.get("n1_n4"),
        )
        additive_gap = None
        if None not in (d12, d23, d34, d14):
            additive_gap = float(d14 - (d12 + d23 + d34))
        nj_row = {
            "item_id": iid,
            "type_key": key,
            "labels": labels,
            "distance_matrix": dmat.tolist(),
            "newick": newick,
            "additive_gap_n1_n4": additive_gap,
            "chain_additive": additive_gap is not None and additive_gap <= atol,
        }
        dist_row["additive_gap_n1_n4"] = additive_gap
        dist_row["newick"] = newick
        if None not in (d12, d23, d14):
            slack = triangle_slack(d12, d23 + (d34 or 0.0), d14)
            tri_rows.append(
                _tri_record(
                    iid,
                    key,
                    "token",
                    "n1",
                    "n2-n3-n4",
                    "n4",
                    d12,
                    (d23 or 0.0) + (d34 or 0.0),
                    d14,
                    atol,
                )
            )
        # SAE chain n1–n2–n4 style using node waypoints
        _append_sae_triangles(tri_rows, iid, key, sae_d, atol)

    elif key == "B":
        # Primary test: trajectory base → shift → combined under disruption.
        d_bs = token_d.get("base_shift")
        d_sc = token_d.get("shift_combined")
        d_bc = token_d.get("base_combined")
        if None not in (d_bs, d_sc, d_bc):
            tri_rows.append(
                _tri_record(
                    iid, key, "token", "base", "shift", "combined", d_bs, d_sc, d_bc, atol
                )
            )
        _append_named_sae_triangle(
            tri_rows, iid, key, sae_d, "base", "shift", "combined", atol
        )

    elif key == "C":
        d_es = token_d.get("entailed_superordinate")
        d_fe = token_d.get("framed_entailed")
        d_fs = token_d.get("framed_superordinate")
        if None not in (d_es, d_fe, d_fs):
            tri_rows.append(
                _tri_record(
                    iid,
                    key,
                    "token",
                    "entailed",
                    "framed",
                    "superordinate",
                    d_fe,
                    d_fs,
                    d_es,
                    atol,
                )
            )
        _append_named_sae_triangle(
            tri_rows,
            iid,
            key,
            sae_d,
            "entailed",
            "framed",
            "superordinate",
            atol,
        )

    n_viol = sum(1 for r in tri_rows if r["violated"])
    dist_row["n_triangle_checks"] = len(tri_rows)
    dist_row["n_triangle_violations"] = n_viol
    return dist_row, tri_rows, nj_row


def math_isfinite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _tri_record(
    item_id: str,
    type_key: str,
    metric: str,
    x: str,
    y: str,
    z: str,
    d_xy: float,
    d_yz: float,
    d_xz: float,
    atol: float,
) -> dict[str, Any]:
    slack = triangle_slack(d_xy, d_yz, d_xz)
    return {
        "item_id": item_id,
        "type_key": type_key,
        "metric": metric,
        "x": x,
        "y": y,
        "z": z,
        "d_xy": d_xy,
        "d_yz": d_yz,
        "d_xz": d_xz,
        "slack": slack,
        "violated": bool(slack > atol),
    }


def _append_named_sae_triangle(
    rows: list[dict[str, Any]],
    iid: str,
    key: str,
    sae_d: dict[str, float],
    a: str,
    b: str,
    c: str,
    atol: float,
) -> None:
    d_ab = sae_d.get(f"sae_{a}_{b}")
    d_bc = sae_d.get(f"sae_{b}_{c}")
    d_ac = sae_d.get(f"sae_{a}_{c}")
    if None in (d_ab, d_bc, d_ac):
        return
    rows.append(_tri_record(iid, key, "sae", a, b, c, d_ab, d_bc, d_ac, atol))


def _append_sae_triangles(
    rows: list[dict[str, Any]],
    iid: str,
    key: str,
    sae_d: dict[str, float],
    atol: float,
) -> None:
    _append_named_sae_triangle(rows, iid, key, sae_d, "node_1", "node_2", "node_4", atol)
    _append_named_sae_triangle(rows, iid, key, sae_d, "node_1", "node_3", "node_4", atol)


def _type_a_distance_matrix(
    token_d: dict[str, float], sae_d: dict[str, float]
) -> np.ndarray:
    """4×4 symmetric matrix over n1..n4. Prefer token d_M; SAE fills gaps."""
    idx = {"n1": 0, "n2": 1, "n3": 2, "n4": 3}
    m = np.full((4, 4), np.nan, dtype=np.float64)
    np.fill_diagonal(m, 0.0)

    def _set(a: str, b: str, value: float | None) -> None:
        if value is None or not np.isfinite(value):
            return
        i, j = idx[a], idx[b]
        m[i, j] = value
        m[j, i] = value

    _set("n1", "n2", token_d.get("n1_n2"))
    _set("n2", "n3", token_d.get("n2_n3"))
    _set("n3", "n4", token_d.get("n3_n4"))
    _set("n1", "n4", token_d.get("n1_n4"))

    node_pairs = [
        ("n1", "n2", "node_1", "node_2"),
        ("n2", "n3", "node_2", "node_3"),
        ("n3", "n4", "node_3", "node_4"),
        ("n1", "n3", "node_1", "node_3"),
        ("n2", "n4", "node_2", "node_4"),
        ("n1", "n4", "node_1", "node_4"),
    ]
    for la, lb, wa, wb in node_pairs:
        i, j = idx[la], idx[lb]
        if not np.isfinite(m[i, j]):
            _set(la, lb, sae_d.get(f"sae_{wa}_{wb}"))

    # Last resort: large finite fill so NJ can still run.
    fill = np.nanmax(m)
    if not np.isfinite(fill):
        fill = 1.0
    m[~np.isfinite(m)] = fill
    return m


def run_analysis(
    *,
    layer_index: int = LAYER_INDEX,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    atol: float = TRIANGLE_ATOL,
    verbose: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    man_path = manifest_jsonl(layer_index, output_dir)
    trop_path = tropical_logprobs_jsonl(layer_index, output_dir)
    items = _load_jsonl(man_path)
    trop_rows = _load_jsonl(trop_path)
    pairs = _pair_map(trop_rows)

    if verbose:
        print(f"manifest={man_path} n_items={len(items)}")
        print(f"tropical_logprobs={trop_path} n_pairs={len(trop_rows)}")
        print(f"layer={layer_index} atol={atol}")

    dist_rows: list[dict[str, Any]] = []
    tri_rows: list[dict[str, Any]] = []
    nj_rows: list[dict[str, Any]] = []

    for item in items:
        drow, tris, nj = analyze_item(
            item,
            pairs.get(str(item["item_id"]), {}),
            layer_index=layer_index,
            output_dir=output_dir,
            atol=atol,
        )
        dist_rows.append(drow)
        tri_rows.extend(tris)
        if nj is not None:
            nj_rows.append(nj)

    dist_path = distances_jsonl(layer_index, output_dir)
    nj_path = nj_newick_jsonl(layer_index, output_dir)
    tri_path = triangle_csv(layer_index, output_dir)
    sum_path = analysis_summary_json(layer_index, output_dir)

    _write_jsonl(dist_path, dist_rows)
    _write_jsonl(nj_path, nj_rows)
    _write_csv(tri_path, tri_rows)

    by_type: dict[str, dict[str, Any]] = {}
    for key in ("A", "B", "C"):
        subset = [r for r in tri_rows if r["type_key"] == key]
        n = len(subset)
        n_viol = sum(1 for r in subset if r["violated"])
        token_sub = [r for r in subset if r["metric"] == "token"]
        sae_sub = [r for r in subset if r["metric"] == "sae"]
        by_type[key] = {
            "n_items": sum(1 for r in dist_rows if r["type_key"] == key),
            "n_triangle_checks": n,
            "n_violations": n_viol,
            "violation_rate": (n_viol / n) if n else None,
            "n_token_violations": sum(1 for r in token_sub if r["violated"]),
            "n_sae_violations": sum(1 for r in sae_sub if r["violated"]),
        }

    additive = [r for r in nj_rows if r.get("additive_gap_n1_n4") is not None]
    summary = {
        "layer_index": layer_index,
        "n_items": len(items),
        "n_distance_rows": len(dist_rows),
        "n_nj_trees": len(nj_rows),
        "n_triangle_checks": len(tri_rows),
        "n_triangle_violations": sum(1 for r in tri_rows if r["violated"]),
        "type_a_mean_additive_gap": (
            float(np.mean([r["additive_gap_n1_n4"] for r in additive]))
            if additive
            else None
        ),
        "type_a_frac_chain_additive": (
            float(np.mean([1.0 if r["chain_additive"] else 0.0 for r in additive]))
            if additive
            else None
        ),
        "by_type": by_type,
        "paths": {
            "distances": str(dist_path),
            "neighbor_joining": str(nj_path),
            "triangle_inequality": str(tri_path),
        },
    }
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(json.dumps(summary, indent=2))
        viol = [r for r in tri_rows if r["violated"]]
        print(f"Triangle-inequality violations: {len(viol)} / {len(tri_rows)}")
        for row in viol[:20]:
            print(
                f"  {row['item_id']} {row['metric']} "
                f"{row['x']}-{row['y']}-{row['z']} slack={row['slack']:.4f}"
            )
        if len(viol) > 20:
            print(f"  … {len(viol) - 20} more")
        print(f"Wrote distances → {dist_path}")
        print(f"Wrote NJ Newick → {nj_path}")
        print(f"Wrote triangle CSV → {tri_path}")
        print(f"Wrote summary → {sum_path}")

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline tropical topology (NJ + triangle inequality) from saved tensors."
    )
    p.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--atol", type=float, default=TRIANGLE_ATOL)
    p.add_argument("-q", "--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_analysis(
        layer_index=args.layer_index,
        output_dir=args.output_dir,
        atol=args.atol,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
