#!/usr/bin/env python3
import argparse
import os
import struct
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import gp_core


def psv_sanity(path):
    size = os.path.getsize(path)
    if size == 0:
        return False, "empty file"
    if size % gp_core.PSV_RECORD_SIZE != 0:
        return False, f"size {size} not multiple of {gp_core.PSV_RECORD_SIZE} (hcpe 等の別形式?)"
    with open(path, "rb") as f:
        raw = f.read(gp_core.PSV_RECORD_SIZE)
    try:
        gp_core.gp_of_record_v2(raw)
    except Exception as e:
        return False, f"first record decode failed: {e}"
    return True, "ok"


def parse_edges(s):
    edges = sorted(float(x) for x in s.split(","))
    if len(edges) < 2:
        sys.exit("[gp] --edges は 2 値以上 (例: 0,0.5,1.51)")
    return edges


def bucket_of(gp, edges):
    for i in range(len(edges) - 1):
        if gp < edges[i + 1] or i == len(edges) - 2:
            return i
    return len(edges) - 2


_W = {}


def _worker(args):
    lo, hi = args
    path, edges, ply_mode, keep_bytes, stride = (
        _W["path"], _W["edges"], _W["ply_mode"], _W["keep"], _W["stride"])
    rs = gp_core.PSV_RECORD_SIZE
    nb = len(edges) - 1
    counts = [0] * nb
    ssum = [0.0] * nb
    psum = [0.0] * nb
    hist = defaultdict(int)
    bufs = [bytearray() for _ in range(nb)] if keep_bytes else None
    bad = 0
    with open(path, "rb") as f:
        for k in range(lo, hi):
            f.seek(k * stride * rs)
            raw = f.read(rs)
            if len(raw) < rs:
                break
            try:
                gp, score, gameply = gp_core.gp_of_record_v2(raw, ply_mode)
            except Exception:
                bad += 1
                continue
            b = bucket_of(gp, edges)
            counts[b] += 1
            ssum[b] += abs(score)
            psum[b] += gameply
            hist[int(gp / 0.05)] += 1
            if bufs is not None:
                bufs[b] += raw
    return counts, ssum, psum, dict(hist), bufs, bad


def _group_one_numba(bin_path, edges, ply_mode, out_dir, stem, report_only,
                     limit=0, writers=None):
    import numpy as np
    gp, sc, pl = gp_core.gp_batch_file(bin_path, ply_mode)
    if limit:
        gp, sc, pl = gp[:limit], sc[:limit], pl[:limit]
    n = gp.shape[0]
    nb = len(edges) - 1
    badm = np.isnan(gp)
    ids = np.clip(np.searchsorted(np.array(edges), gp, side="right") - 1, 0, nb - 1)
    if not report_only and writers is not None:
        recs = np.fromfile(bin_path, dtype=np.uint8)[:n * 40].reshape(n, 40)
        for i in range(nb):
            sel = (ids == i) & ~badm
            if sel.any():
                writers[i].write(recs[sel].tobytes())
        del recs
    good = ~badm
    counts = [0] * nb
    ssum = [0.0] * nb
    psum = [0.0] * nb
    hc = np.zeros(30, np.int64)
    for i in range(nb):
        sel = (ids == i) & good
        c = int(sel.sum())
        counts[i] = c
        if c:
            ssum[i] = float(np.abs(sc[sel]).sum())
            psum[i] = float(pl[sel].sum())
    if good.any():
        hbins = np.clip((gp[good] / 0.05).astype(np.int64), 0, 29)
        hc += np.bincount(hbins, minlength=30)
    return {"counts": counts, "ssum": ssum, "psum": psum,
            "n": n, "bad": int(badm.sum()), "hc": hc}


def main():
    ap = argparse.ArgumentParser(description="PSV 教師局面を GP でグルーピング")
    ap.add_argument("bin", help="PackedSfenValue .bin (40B/レコード) またはディレクトリ")
    ap.add_argument("--edges", default=",".join(f"{x/10:.1f}" for x in range(16)))
    ap.add_argument("--ply-mode", default="auto", choices=gp_core.PLY_MODES,
                    help="ply の扱い (docstring 参照; dedup 教師は state 推奨)")
    ap.add_argument("--lane", default="auto", choices=("auto", "numba", "py"),
                    help="auto: numba があればバッチレーン (既定)。--sample 時は py")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="並列プロセス数 (既定: 全コア)")
    ap.add_argument("--sample", type=int, default=0,
                    help="等間隔に N 局面だけ処理 (分布の下見用)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--require-done", action="store_true",
                    help="ディレクトリ指定時、.done マーカー付き bin のみ処理")
    ap.add_argument("--split-bytes", type=int, default=gp_core._DEFAULT_SPLIT_BYTES,
                    help="バケット bin を N bytes 単位で分割 (0=無効, 既定≈19GB)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if os.path.isdir(args.bin):
        bin_files = gp_core.expand_bin_paths_done(args.bin, require_done=args.require_done)
        if not bin_files:
            sys.exit(f"[gp] .done 付き .bin が見つかりません: {args.bin}")
        stem = os.path.basename(os.path.abspath(args.bin).rstrip("/\\")) or "group"
    else:
        bin_files = [args.bin]
        stem = os.path.splitext(os.path.basename(args.bin))[0]

    ok, msg = psv_sanity(bin_files[0])
    if not ok:
        sys.exit(f"[gp] psv sanity NG: {msg}")

    edges = parse_edges(args.edges)
    nb = len(edges) - 1
    out_dir = args.out_dir or os.path.join(".", f"{stem}_gp")
    os.makedirs(out_dir, exist_ok=True)

    if args.lane != "py" and gp_core.NUMBA_OK and not args.sample:
        import numpy as np
        t0 = time.perf_counter()
        writers = {}
        if not args.report_only:
            for i in range(nb):
                name = os.path.join(out_dir, f"{stem}_gp_{edges[i]:.2f}-{edges[i+1]:.2f}.bin")
                writers[i] = gp_core.ChunkedBinWriter(name, max_bytes=args.split_bytes)
        total_counts = [0] * nb
        total_ssum = [0.0] * nb
        total_psum = [0.0] * nb
        total_n = 0
        total_bad = 0
        total_hc = np.zeros(30, np.int64)
        for fi, bf in enumerate(bin_files):
            print(f"[gp] ({fi+1}/{len(bin_files)}) {os.path.basename(bf)} ...", flush=True)
            st = _group_one_numba(bf, edges, args.ply_mode, out_dir, stem,
                                  args.report_only, args.limit,
                                  writers if writers else None)
            for i in range(nb):
                total_counts[i] += st["counts"][i]
                total_ssum[i] += st["ssum"][i]
                total_psum[i] += st["psum"][i]
            total_n += st["n"]
            total_bad += st["bad"]
            total_hc += st["hc"]
        for w in writers.values():
            w.close()
        el = time.perf_counter() - t0
        lines = [f"# GP grouping report — {args.bin} ({len(bin_files)} files)",
                 f"# gp=gp_core v2 [numba lane] ply_mode={args.ply_mode} edges={edges} "
                 f"処理={total_n} 不良={total_bad} ({el:.2f}s, {total_n/max(1e-9,el):,.0f} pos/s)",
                 ""]
        lines.append(f"{'bucket':22s} {'count':>10s} {'%':>7s} {'avg|cp|':>9s} {'avg ply':>8s}")
        total_good = total_n - total_bad
        for i in range(nb):
            c = total_counts[i]
            pct = 100.0 * c / max(1, total_good)
            asc = total_ssum[i] / c if c else 0.0
            apl = total_psum[i] / c if c else 0.0
            lines.append(f"GP[{edges[i]:.2f},{edges[i+1]:.2f}) {c:>10d} {pct:>6.2f}% "
                         f"{asc:>9.1f} {apl:>8.1f}")
        lines.append("")
        lines.append("# 0.05 刻みヒストグラム")
        mx = max(1, int(total_hc.max()))
        for k in range(30):
            if total_hc[k]:
                lines.append(f"{k*0.05:5.2f} {int(total_hc[k]):>9d} " + "#" * max(1, int(40 * total_hc[k] / mx)))
        report = "\n".join(lines)
        print(report)
        rp = os.path.join(out_dir, f"{stem}_gp_report.txt")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"[gp] report → {rp}")
        if not args.report_only:
            print(f"[gp] buckets → {out_dir}/{stem}_gp_*.bin")
        return

    if len(bin_files) > 1:
        sys.exit("[gp] Python レーンはディレクトリ非対応。numba を導入するか "
                 "個別ファイルを指定してください")
    bin_path = bin_files[0]
    rs = gp_core.PSV_RECORD_SIZE
    total = os.path.getsize(bin_path) // rs

    stride, todo = 1, total
    if args.sample and args.sample < total:
        stride = total // args.sample
        todo = args.sample
    if args.limit:
        todo = min(todo, args.limit)

    keep = not args.report_only
    _W.update(path=os.path.abspath(bin_path), edges=edges,
              ply_mode=args.ply_mode, keep=keep, stride=stride)
    nw = max(1, min(args.workers, todo))
    chunk = (todo + nw - 1) // nw
    ranges = [(i * chunk, min((i + 1) * chunk, todo)) for i in range(nw)
              if i * chunk < todo]

    t0 = time.perf_counter()
    if nw == 1:
        results = [_worker(ranges[0])]
    else:
        with Pool(nw) as pool:
            results = pool.map(_worker, ranges)
    nb = len(edges) - 1
    counts = [0] * nb
    ssum = [0.0] * nb
    psum = [0.0] * nb
    hist = defaultdict(int)
    bad = 0
    writers = {}
    if keep:
        for i in range(nb):
            name = os.path.join(out_dir, f"{stem}_gp_{edges[i]:.2f}-{edges[i+1]:.2f}.bin")
            writers[i] = gp_core.ChunkedBinWriter(name, max_bytes=args.split_bytes)
    for c, s, p, h, bufs, bd in results:
        for i in range(nb):
            counts[i] += c[i]; ssum[i] += s[i]; psum[i] += p[i]
            if writers and bufs is not None:
                writers[i].write(bufs[i])
        for k, v in h.items():
            hist[k] += v
        bad += bd
    for w in writers.values():
        w.close()
    el = time.perf_counter() - t0
    done = sum(counts) + bad

    lines = [f"# GP grouping report — {bin_path}",
             f"# gp=gp_core v2 ply_mode={args.ply_mode} workers={nw} edges={edges} "
             f"処理={done} 不良={bad} ({el:.1f}s, {done/max(1e-9,el):.0f} pos/s)",
             ""]
    lines.append(f"{'bucket':22s} {'count':>10s} {'%':>7s} {'avg|cp|':>9s} {'avg ply':>8s}")
    for i in range(nb):
        c = counts[i]
        pct = 100.0 * c / max(1, done - bad)
        lines.append(f"GP[{edges[i]:.2f},{edges[i+1]:.2f}) {c:>10d} {pct:>6.2f}% "
                         f"{(ssum[i]/c if c else 0):>9.1f} {(psum[i]/c if c else 0):>8.1f}")
    lines.append("")
    lines.append("# 0.05 刻みヒストグラム")
    mx = max(hist.values()) if hist else 1
    for k in sorted(hist):
        lines.append(f"{k*0.05:5.2f} {hist[k]:>9d} " + "#" * max(1, int(40 * hist[k] / mx)))
    report = "\n".join(lines)
    print(report)
    rp = os.path.join(out_dir, f"{stem}_gp_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[gp] report → {rp}")
    if writers:
        print(f"[gp] buckets → {out_dir}/{stem}_gp_*.bin")


if __name__ == "__main__":
    main()
