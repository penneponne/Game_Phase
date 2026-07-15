#!/usr/bin/env python3
"""gp_group_psv.py — PackedSfenValue (.bin) 教師局面を GamePhase (GP) でグルーピング v2

Pyfamate (https://github.com/SH11235/pyfamate) の GP 計算モジュール。
PackedSfenValue 40B フォーマットは rshogi gensfen / tatara / bullet-shogi と
完全互換 — これらで生成した教師 .bin をそのまま入力できる。
  - rshogi   : https://github.com/SH11235/rshogi   (gensfen / shuffle / rescore)
  - tatara   : https://github.com/SH11235/tatara    (NNUE GPU 学習)
  - bullet-shogi: https://github.com/SH11235/bullet-shogi (NNUE 学習, 旧)

  GP でバケット分割した .bin は tatara nnue-train (--data) にそのまま渡せる。
  ⚠ tatara progress-kpabs-train は対局順の連続 PSV が必須のため、GP 分割後の
  データは progress 学習には使用不可。progress.bin は分割前のデータで生成すること。

GP 計算は gp_core.py (Pyfamate 本体から原文抽出した独立 GP 計算部)。
60K 行本体の import なしで動く。等価性検証: `python gp_core.py --verify <bin>` /
高速路検証: `python gp_core.py --verify-fast <bin>` (139,128局面で mismatch 0 確認済)。

  ply の扱い (--ply-mode, 既定 auto):
    record : レコードの gamePly をそのまま (live 経路と同一式)
    zero   : ply 項なし (純状態GP)。⚠ 実測 mean|ΔGP|=0.167・バケット移動61.6% —
             「ply が無い」を 0 で表すと系統バイアスになる。単独使用は非推奨
    state  : 盤面状態からの期待 ply (較正テーブル) — dedup 教師の「初出対局の
             ply」ノイズを平滑化。record 比 mean|ΔGP|=0.035・移動13.9%
    auto   : gamePly>=1 なら record、0(欠損) なら state — ply 欠損がノイズに
             ならない既定
  dedup 済み教師 (unique 系) のグルーピング安定性を最優先するなら
  --ply-mode state を推奨 (記録 ply 自体が per-game ノイズ ±0.03GP を持つため)。

  速度: v2 高速路 (SFEN 往復排除) ~39k pos/s/コア + --workers 並列。
    16 コア機なら ~500k pos/s 見込み。さらに上 (数億局面) は gp_core の
    ステータス md 記載の numba/GPU/Rust 移植パスを参照。

  使い方:
    python gp_group_psv.py teacher.bin [--ply-mode state] [--workers 16]
    python gp_group_psv.py teacher.bin --edges 0,0.25,0.5,0.75,1.0,1.51
    python gp_group_psv.py teacher.bin --sample 20000 --report-only
"""
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


_W = {}   # worker 初期化パラメータ (fork 継承)


def _worker(args):
    """レコード範囲 [lo, hi) を処理し (counts, score_sum, ply_sum, hist, bucket_bytes, bad) を返す。"""
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
    """numba レーンで 1 ファイルをグルーピング。累積統計 dict を返す。
    writers が渡された場合はバケットファイルに append。"""
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

    # ── ディレクトリ指定: .done マーカー付き bin を順次処理 ──
    if os.path.isdir(args.bin):
        bin_files = gp_core.expand_bin_paths_done(args.bin, require_done=args.require_done)
        if not bin_files:
            sys.exit(f"[gp] .done 付き .bin が見つかりません: {args.bin}")
        # フォルダ名を stem に
        stem = os.path.basename(os.path.abspath(args.bin).rstrip("/\\")) or "group"
    else:
        bin_files = [args.bin]
        stem = os.path.splitext(os.path.basename(args.bin))[0]

    # 最初のファイルで sanity check
    ok, msg = psv_sanity(bin_files[0])
    if not ok:
        sys.exit(f"[gp] psv sanity NG: {msg}")

    edges = parse_edges(args.edges)
    nb = len(edges) - 1
    out_dir = args.out_dir or (os.path.abspath(args.bin)
                               if os.path.isdir(args.bin)
                               else os.path.dirname(os.path.abspath(args.bin)))
    os.makedirs(out_dir, exist_ok=True)

    # ── numba バッチレーン ──
    if args.lane != "py" and gp_core.NUMBA_OK and not args.sample:
        import numpy as np
        t0 = time.perf_counter()
        writers = {}
        if not args.report_only:
            for i in range(nb):
                name = os.path.join(out_dir, f"{stem}_gp_{edges[i]:.2f}-{edges[i+1]:.2f}.bin")
                writers[i] = gp_core.ChunkedBinWriter(name, max_bytes=args.split_bytes)
        # 統計の累積
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

    # ── Python (マルチプロセス) レーン — 単一ファイルのみ ──
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