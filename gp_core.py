#!/usr/bin/env python3
import math
import os
import struct
import sys

PSV_RECORD_SIZE = 40
PSV_FMT = "<32shHHbB"


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


_HUF_BOARD = [
    (0x00, 1),
    (0x01, 2),
    (0x03, 4),
    (0x0b, 4),
    (0x07, 4),
    (0x1f, 6),
    (0x3f, 6),
    (0x0f, 5),
]

_HUF_HAND_MATCH = [(code >> 1, bits - 1) for code, bits in _HUF_BOARD]


_PSFEN_SQ_MAP = tuple((sq % 9) * 9 + 8 - (sq // 9) for sq in range(81))

_PT_BASE_PIDX  = (0, 0, 1, 2, 3, 5, 6, 4, 7)
_PT_PROM_PIDX  = (0, 8, 9, 10, 11, 12, 13)
_PT_HAND_IDX   = (0, 0, 1, 2, 3, 5, 6, 4)

_FAST_BOARD_LUT = [None] * 256
for _v in range(256):
    for _pt in range(8):
        _c, _b = _HUF_BOARD[_pt]
        if (_v & ((1 << _b) - 1)) == _c:
            if _pt == 0:
                _FAST_BOARD_LUT[_v] = (0, _b)
            elif _pt == 7:
                _FAST_BOARD_LUT[_v] = (1 + 4 + ((_v >> _b) & 1) * 14, _b + 1)
            else:
                _pm = (_v >> _b) & 1
                _col = (_v >> (_b + 1)) & 1
                _pi = _PT_PROM_PIDX[_pt] if _pm else _PT_BASE_PIDX[_pt]
                _FAST_BOARD_LUT[_v] = (1 + _pi + _col * 14, _b + 2)
            break
_FAST_BOARD_LUT = tuple(_FAST_BOARD_LUT)

_FAST_HAND_LUT = [None] * 128
for _v in range(128):
    for _pt in range(1, 8):
        _c, _b = _HUF_HAND_MATCH[_pt]
        if (_v & ((1 << _b) - 1)) == _c:
            if _pt == 7:
                _FAST_HAND_LUT[_v] = (_PT_HAND_IDX[_pt], (_v >> _b) & 1, _b + 1)
            else:
                _pm = (_v >> _b) & 1
                _col = (_v >> (_b + 1)) & 1
                _FAST_HAND_LUT[_v] = (-1 if _pm else _PT_HAND_IDX[_pt], _col, _b + 2)
            break
_FAST_HAND_LUT = tuple(_FAST_HAND_LUT)


def _fast_decode_psfen_into(data32, sb):
    bint = int.from_bytes(data32, 'little')
    cells = sb.cells
    hand0 = sb.hand[0]
    hand1 = sb.hand[1]
    sb.side = bint & 1
    cur = 1
    k0 = (bint >> 1) & 0x7F;  cur = 8
    k1 = (bint >> 8) & 0x7F;  cur = 15
    sqm = _PSFEN_SQ_MAP
    if k0 < 81:
        cells[sqm[k0]] = 8
    if k1 < 81:
        cells[sqm[k1]] = 22
    blut = _FAST_BOARD_LUT
    for sq in range(81):
        if sq == k0 or sq == k1:
            continue
        v = (bint >> cur) & 0xFF
        e = blut[v]
        if e is None:
            raise IndexError("psfen Huffman overrun bit %d" % cur)
        cells[sqm[sq]] = e[0]
        cur += e[1]
    hlut = _FAST_HAND_LUT
    while cur < 256:
        v = (bint >> cur) & 0x7F
        e = hlut[v]
        if e is None:
            raise IndexError("psfen hand overrun bit %d" % cur)
        if e[0] >= 0:
            if e[1] == 0:
                hand0[e[0]] += 1
            else:
                hand1[e[0]] += 1
        cur += e[2]


_PID_SFEN = (None,
    "P", "L", "N", "S", "G", "B", "R", "K",
    "+P", "+L", "+N", "+S", "+B", "+R",
    "p", "l", "n", "s", "g", "b", "r", "k",
    "+p", "+l", "+n", "+s", "+b", "+r")


def _svboard_to_sfen(sb):
    cells = sb.cells
    ranks = []
    for r in range(9):
        row = []; empty = 0
        base = r * 9
        for c in range(9):
            pid = cells[base + c]
            if pid == 0:
                empty += 1
            else:
                if empty:
                    row.append(str(empty)); empty = 0
                row.append(_PID_SFEN[pid])
        if empty:
            row.append(str(empty))
        ranks.append("".join(row))
    board_str = "/".join(ranks)
    turn = "b" if sb.side == 0 else "w"
    hp = []
    for owner in range(2):
        h = sb.hand[owner]
        for ki in (6, 5, 4, 3, 2, 1, 0):
            n = h[ki]
            if n > 0:
                ch = "PLNSGBR"[ki]
                if owner == 1: ch = ch.lower()
                hp.append(("" if n == 1 else str(n)) + ch)
    hand_str = "".join(hp) or "-"
    return "%s %s %s %d" % (board_str, turn, hand_str, sb.ply)


def _fast_psfen_to_sfen(data32):
    sb = SvBoard()
    _fast_decode_psfen_into(data32, sb)
    return _svboard_to_sfen(sb)


GP_HAND_VALS = [1.0,    2.9878, 2.9303, 4.8517, 4.915,  7.7843, 9.8869]
GP_CAMP_VALS = [1.0601, 1.0085, 1.016,  1.0546, 1.0494, 1.0528, 1.0605,
                0.9574, 1.0072, 1.0054, 1.0159, 1.031,  1.0719]
CAMP_TYPE_NAMES = ("P", "L", "N", "S", "G", "B", "R",
                   "+P", "+L", "+N", "+S", "+B", "+R")
_GAME_PHASE_KING_SENTE_ROW = 7
_GAME_PHASE_KING_GOTE_ROW  = 3

_GAME_PHASE_DEFAULT_VEC = (
    0.266368, 0.0, 0.275166, 0.352129,
    1.839152, 5.437156, 1.000000, 0.100000,
    1.285300,
    0.531332, 4.362364,
    0.074838, 163.738998, 0.0, 12.181117,
)


class _GpParamView:
    __slots__ = (
        "game_phase_w_hand", "game_phase_w_promo", "game_phase_w_camp", "game_phase_w_king",
        "game_phase_hand_ref", "game_phase_promo_ref", "game_phase_camp_ref", "game_phase_king_ref",
        "game_phase_scale", "game_phase_nyugyoku_king_onset", "game_phase_nyugyoku_king_full",
        "game_phase_w_ply", "game_phase_ply_ref", "game_phase_w_exposure", "game_phase_exposure_ref",
        "game_phase_ply_enable",
    )

    def __init__(self, vec):
        (self.game_phase_w_hand, self.game_phase_w_promo, self.game_phase_w_camp, self.game_phase_w_king,
         self.game_phase_hand_ref, self.game_phase_promo_ref, self.game_phase_camp_ref, self.game_phase_king_ref,
         self.game_phase_scale, self.game_phase_nyugyoku_king_onset,
         self.game_phase_nyugyoku_king_full,
         self.game_phase_w_ply, self.game_phase_ply_ref,
         self.game_phase_w_exposure, self.game_phase_exposure_ref) = vec
        self.game_phase_ply_enable = True


def _calc_state_phase(WHAND: int, PROMO: int, CAMP: int, KING: float,
                      EXPOSURE: int, *, cfg, ply: int = 0) -> float:
    _w_hand    = cfg.game_phase_w_hand
    _w_promo   = cfg.game_phase_w_promo
    _w_camp    = cfg.game_phase_w_camp
    _w_king    = cfg.game_phase_w_king
    _hand_ref  = cfg.game_phase_hand_ref
    _promo_ref = cfg.game_phase_promo_ref
    _camp_ref  = cfg.game_phase_camp_ref
    _king_ref  = cfg.game_phase_king_ref
    _scale     = cfg.game_phase_scale
    _w_ply     = cfg.game_phase_w_ply
    _ply_ref   = cfg.game_phase_ply_ref
    _w_expo    = cfg.game_phase_w_exposure
    _expo_ref  = cfg.game_phase_exposure_ref
    _ply_term = (_w_ply * ply / _ply_ref) if cfg.game_phase_ply_enable else 0.0
    S = (_w_hand  * WHAND / _hand_ref +
         _w_promo * PROMO / _promo_ref +
         _w_camp  * CAMP  / _camp_ref +
         _w_king  * KING  / _king_ref +
         _ply_term +
         _w_expo  * EXPOSURE / _expo_ref)
    neg_s_over_scale = -S / _scale
    if neg_s_over_scale < -10.0:
        game_phase_core = 1.0
    else:
        game_phase_core = 1.0 - math.exp(neg_s_over_scale)

    _onset = cfg.game_phase_nyugyoku_king_onset
    _full  = cfg.game_phase_nyugyoku_king_full
    nyugyoku_bonus = 0.0
    if KING > _onset:
        frac = (KING - _onset) / (_full - _onset)
        if frac > 1.0:
            frac = 1.0
        nyugyoku_bonus = 0.5 * frac

    game_phase = game_phase_core + nyugyoku_bonus
    return _clamp(game_phase, 0.0, 1.5)


_GAME_PHASE_NYUGYOKU_BAND_WIDTH = 0.5


def _game_phase_for_phase_council(raw_game_phase: float, restore_ratio: float) -> float:
    if raw_game_phase is None:
        return 0.0
    if raw_game_phase <= 1.0 or restore_ratio <= 0.0:
        return min(max(raw_game_phase, 0.0), 1.0)
    frac = (raw_game_phase - 1.0) / _GAME_PHASE_NYUGYOKU_BAND_WIDTH
    if frac > 1.0:
        frac = 1.0
    return max(0.0, 1.0 - restore_ratio * frac)


def _game_phase_state_counts_core(board: "ShogiBoard") -> tuple:
    _HAND_IDX = {"P": 0, "L": 1, "N": 2, "S": 3, "G": 4, "B": 5, "R": 6}
    whand = 0.0
    for hand in (board.sente_hand, board.gote_hand):
        for pc, cnt in hand.items():
            whand += cnt * GP_HAND_VALS[_HAND_IDX[pc.upper()]]
    _CAMP_IDX = {"P": 0, "L": 1, "N": 2, "S": 3, "G": 4, "B": 5, "R": 6,
                 "+P": 7, "+L": 8, "+N": 9, "+S": 10, "+B": 11, "+R": 12}
    promoted = 0
    wcamp = 0.0
    king_positions = {}
    cells = []
    for (c, r), pc in board.board.items():
        if pc.startswith("+"):
            promoted += 1
        base = pc.lstrip("+")
        if base == "K":
            king_positions["b"] = (c, r)
            continue
        elif base == "k":
            king_positions["w"] = (c, r)
            continue
        is_sente = base.isupper()
        cells.append((c, r, is_sente))
        if (is_sente and r <= 3) or ((not is_sente) and r >= 7):
            _t = ("+" if pc.startswith("+") else "") + base.upper()
            wcamp += GP_CAMP_VALS[_CAMP_IDX[_t]]
    ally_ring1 = 0
    for side, is_ally_sente in [("b", True), ("w", False)]:
        kp = king_positions.get(side)
        if kp is None:
            continue
        kc, kr = kp
        for cc, cr, piece_sente in cells:
            if piece_sente == is_ally_sente and max(abs(cc - kc), abs(cr - kr)) == 1:
                ally_ring1 += 1
    exposure = 16 - ally_ring1
    return (whand, promoted, wcamp, board.king_advance_signal(), exposure)




class SvBoard:
    def __init__(self):
        self.cells = [0] * 81
        self.hand = [[0] * 7, [0] * 7]
        self.side = 0
        self.ply = 1


class MiniBoard:
    def __init__(self, sfen_body: str):
        board_part, turn, hand_part = (sfen_body.split() + ["-"])[:3]
        self.board = {}
        self.sente_hand = {}
        self.gote_hand = {}
        self._kpos = [None, None]
        for r_idx, row in enumerate(board_part.split("/")):
            rank = r_idx + 1
            file = 9
            i = 0
            while i < len(row):
                ch = row[i]
                if ch.isdigit():
                    file -= int(ch)
                    i += 1
                    continue
                if ch == "+":
                    pc = ch + row[i + 1]
                    i += 2
                else:
                    pc = ch
                    i += 1
                self.board[(file, rank)] = pc
                file -= 1
        if hand_part != "-":
            i = 0
            while i < len(hand_part):
                n = 0
                while hand_part[i].isdigit():
                    n = n * 10 + int(hand_part[i])
                    i += 1
                n = n or 1
                ch = hand_part[i]
                i += 1
                tgt = self.sente_hand if ch.isupper() else self.gote_hand
                tgt[ch] = tgt.get(ch, 0) + n

    def _king_pos(self, side: str):
        _idx = 0 if side == "b" else 1
        _kp = self._kpos[_idx]
        if _kp is not None:
            return _kp
        target = "K" if side == "b" else "k"
        for pos, pc in self.board.items():
            if pc == target:
                self._kpos[_idx] = pos
                return pos
        return None

    def king_advance_signal(self) -> float:
        total = 0.0
        for side, entry, ahead in (("b", _GAME_PHASE_KING_SENTE_ROW, -1),
                                   ("w", _GAME_PHASE_KING_GOTE_ROW, 1)):
            kp = self._king_pos(side)
            if kp is None:
                continue
            kc, kr = kp
            raw = (entry - kr) if ahead < 0 else (kr - entry)
            if raw < 0:
                continue
            sup = 0
            for (c, r), pc in self.board.items():
                base = pc.lstrip("+")
                if base in ("K", "k"):
                    continue
                is_sente = base.isupper()
                if is_sente != (side == "b"):
                    continue
                if max(abs(c - kc), abs(r - kr)) <= 2 and (
                        r <= kr + 1 if ahead < 0 else r >= kr - 1):
                    sup += 1
            if sup > 3:
                sup = 3
            total += (raw + 0.5) * (0.25 + 0.25 * sup)
        return total




def features_from_cells(sb):
    cells = sb.cells
    h0, h1 = sb.hand
    h7 = tuple(h0[k] + h1[k] for k in range(7))
    c13 = [0] * 13
    promoted = 0
    ks = kw = None
    sente_sq = []
    gote_sq = []
    for sq in range(81):
        pid = cells[sq]
        if pid == 0:
            continue
        r = sq // 9
        c = sq % 9
        if pid == 8:
            ks = (r, c); continue
        if pid == 22:
            kw = (r, c); continue
        base = (pid - 1) % 14 + 1
        t = (base - 1) if base <= 7 else (7 + base - 9)
        is_sente = pid <= 14
        if base >= 9:
            promoted += 1
        if is_sente:
            sente_sq.append((r, c))
            if r <= 2:
                c13[t] += 1
        else:
            gote_sq.append((r, c))
            if r >= 6:
                c13[t] += 1
    ally_ring1 = 0
    if ks is not None:
        kr, kc = ks
        for r, c in sente_sq:
            if max(abs(r - kr), abs(c - kc)) == 1:
                ally_ring1 += 1
    if kw is not None:
        kr, kc = kw
        for r, c in gote_sq:
            if max(abs(r - kr), abs(c - kc)) == 1:
                ally_ring1 += 1
    exposure = 16 - ally_ring1
    king = 0.0
    if ks is not None:
        raw = _GAME_PHASE_KING_SENTE_ROW - (ks[0] + 1)
        if raw >= 0:
            sup = 0
            for r, c in sente_sq:
                if (max(abs(r - ks[0]), abs(c - ks[1])) <= 2
                        and r <= ks[0] + 1):
                    sup += 1
            if sup > 3:
                sup = 3
            king += (raw + 0.5) * (0.25 + 0.25 * sup)
    if kw is not None:
        raw = (kw[0] + 1) - _GAME_PHASE_KING_GOTE_ROW
        if raw >= 0:
            sup = 0
            for r, c in gote_sq:
                if (max(abs(r - kw[0]), abs(c - kw[1])) <= 2
                        and r >= kw[0] - 1):
                    sup += 1
            if sup > 3:
                sup = 3
            king += (raw + 0.5) * (0.25 + 0.25 * sup)
    return h7, tuple(c13), promoted, king, exposure



_PLY_CALIB = (42.0, 59.0, 65.0, 67.0, 83.0, 93.0, 100.0, 107.0, 117.0, 120.0, 120.0, 119.0, 120.0, 125.0, 127.0, 132.0, 131.0, 134.0, 140.0, 149.0, 164.0, 186.0, 229.0, 240.0, 264.0, 282.0)


def estimate_ply_from_state(gp0: float) -> float:
    t = _PLY_CALIB
    if not t:
        return 0.0
    x = gp0 / 0.05
    k = int(x)
    if k >= len(t) - 1:
        return float(t[-1])
    f = x - k
    return t[k] * (1.0 - f) + t[k + 1] * f


try:
    if getattr(sys, "frozen", False):
        os.environ.setdefault(
            "NUMBA_CACHE_DIR",
            os.path.join(os.environ.get("TEMP", "/tmp"), "gp_numba_cache"))
    import numpy as _np
    from numba import njit as _njit, prange as _prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

if NUMBA_OK:
    _NB_BOARD = _np.full((256, 2), -1, _np.int16)
    for _v, _e in enumerate(_FAST_BOARD_LUT):
        if _e is not None:
            _NB_BOARD[_v, 0], _NB_BOARD[_v, 1] = _e
    _NB_HAND = _np.full((128, 3), -1, _np.int16)
    for _v, _e in enumerate(_FAST_HAND_LUT):
        if _e is not None:
            _NB_HAND[_v] = _e
    _NB_SQMAP = _np.array(_PSFEN_SQ_MAP, _np.int16)
    _NB_HANDW = _np.array((1, 3, 3, 5, 5, 8, 10), _np.int16)

    @_njit(cache=True, parallel=True)
    def _nb_gp_all(buf, n, board_lut, hand_lut, sqmap, hv, cv, calib,
                   wh, wpm, wc, wk, hr, pr, cr, kr, scale, onset, full,
                   wply, plyref, wexpo, exporef, mode):
        out = _np.empty(n, _np.float64)
        scores = _np.empty(n, _np.int32)
        plys = _np.empty(n, _np.int32)
        for rec in _prange(n):
            base = rec * 40
            ok = True
            cells = _np.zeros(81, _np.int8)
            h = _np.zeros(14, _np.int16)
            side = buf[base] & 1
            b0 = _np.uint32(buf[base]) | (_np.uint32(buf[base + 1]) << 8)
            k0 = (b0 >> 1) & 0x7F
            k1 = (b0 >> 8) & 0x7F
            cur = 15
            if k0 < 81:
                cells[sqmap[k0]] = 8
            if k1 < 81:
                cells[sqmap[k1]] = 22
            for sq in range(81):
                if sq == k0 or sq == k1:
                    continue
                byi = cur >> 3
                lo = _np.uint32(buf[base + byi])
                if byi + 1 < 32:
                    lo |= _np.uint32(buf[base + byi + 1]) << 8
                v = (lo >> (cur & 7)) & 0xFF
                pid = board_lut[v, 0]
                if pid < 0:
                    ok = False
                    break
                cells[sqmap[sq]] = pid
                cur += board_lut[v, 1]
            if ok:
                while cur < 256:
                    byi = cur >> 3
                    lo = _np.uint32(buf[base + byi])
                    if byi + 1 < 32:
                        lo |= _np.uint32(buf[base + byi + 1]) << 8
                    v = (lo >> (cur & 7)) & 0x7F
                    hidx = hand_lut[v, 0]
                    hcol = hand_lut[v, 1]
                    hb = hand_lut[v, 2]
                    if hb < 0:
                        ok = False
                        break
                    if hidx >= 0:
                        h[hcol * 7 + hidx] += 1
                    cur += hb
            if not ok:
                out[rec] = _np.nan
                scores[rec] = 0
                plys[rec] = 0
                continue
            s = _np.int32(buf[base + 32]) | (_np.int32(buf[base + 33]) << 8)
            if s >= 32768:
                s -= 65536
            gameply = _np.int32(buf[base + 36]) | (_np.int32(buf[base + 37]) << 8)
            scores[rec] = s
            plys[rec] = gameply
            whand = 0.0
            for k in range(7):
                whand += (h[k] + h[7 + k]) * hv[k]
            wcamp = 0.0
            promoted = 0
            ksr = -1; ksc = -1; kwr = -1; kwc = -1
            for sq in range(81):
                pid = cells[sq]
                if pid == 0:
                    continue
                r = sq // 9
                c = sq % 9
                if pid == 8:
                    ksr = r; ksc = c
                elif pid == 22:
                    kwr = r; kwc = c
                else:
                    base = (pid - 1) % 14 + 1
                    t = (base - 1) if base <= 7 else (7 + base - 9)
                    if base >= 9:
                        promoted += 1
                    if pid <= 14:
                        if r <= 2:
                            wcamp += cv[t]
                    else:
                        if r >= 6:
                            wcamp += cv[t]
            ring1 = 0
            for sq in range(81):
                pid = cells[sq]
                if pid == 0 or pid == 8 or pid == 22:
                    continue
                r = sq // 9
                c = sq % 9
                if pid <= 14:
                    if ksr >= 0 and max(abs(r - ksr), abs(c - ksc)) == 1:
                        ring1 += 1
                else:
                    if kwr >= 0 and max(abs(r - kwr), abs(c - kwc)) == 1:
                        ring1 += 1
            exposure = 16 - ring1
            king = 0.0
            if ksr >= 0:
                raw_k = 7 - (ksr + 1)
                if raw_k >= 0:
                    sup = 0
                    for sq in range(81):
                        pid = cells[sq]
                        if pid == 0 or pid == 8 or pid > 14:
                            continue
                        r = sq // 9
                        c = sq % 9
                        if (max(abs(r - ksr), abs(c - ksc)) <= 2
                                and r <= ksr + 1):
                            sup += 1
                    if sup > 3:
                        sup = 3
                    king += (raw_k + 0.5) * (0.25 + 0.25 * sup)
            if kwr >= 0:
                raw_k = (kwr + 1) - 3
                if raw_k >= 0:
                    sup = 0
                    for sq in range(81):
                        pid = cells[sq]
                        if pid <= 14 or pid == 22:
                            continue
                        r = sq // 9
                        c = sq % 9
                        if (max(abs(r - kwr), abs(c - kwc)) <= 2
                                and r >= kwr - 1):
                            sup += 1
                    if sup > 3:
                        sup = 3
                    king += (raw_k + 0.5) * (0.25 + 0.25 * sup)
            S0 = (wh * whand / hr + wpm * promoted / pr + wc * wcamp / cr +
                  wk * king / kr + wexpo * exposure / exporef)
            bonus = 0.0
            if king > onset:
                frac = (king - onset) / (full - onset)
                if frac > 1.0:
                    frac = 1.0
                bonus = 0.5 * frac
            def_core = 0.0
            if mode == 0:
                ply = float(gameply)
            elif mode == 1:
                ply = 0.0
            else:
                z0 = -S0 / scale
                c0 = 1.0 if z0 < -10.0 else 1.0 - _np.exp(z0)
                gp0 = c0 + bonus
                if gp0 < 0.0:
                    gp0 = 0.0
                if gp0 > 1.5:
                    gp0 = 1.5
                if mode == 3 and gameply >= 1:
                    ply = float(gameply)
                else:
                    x = gp0 / 0.05
                    k2 = int(x)
                    if k2 >= calib.shape[0] - 1:
                        ply = calib[calib.shape[0] - 1]
                    else:
                        fca = x - k2
                        ply = calib[k2] * (1.0 - fca) + calib[k2 + 1] * fca
            S = S0 + (wply * ply / plyref)
            z = -S / scale
            core = 1.0 if z < -10.0 else 1.0 - _np.exp(z)
            gp = core + bonus
            if gp < 0.0:
                gp = 0.0
            if gp > 1.5:
                gp = 1.5
            out[rec] = gp
        return out, scores, plys

    @_njit(cache=True, parallel=True)
    def _nb_feat_all(buf, n, board_lut, hand_lut, sqmap):
        feat = _np.zeros((n, 23), _np.float32)
        plys = _np.empty(n, _np.int32)
        scores = _np.empty(n, _np.int32)
        for rec in _prange(n):
            base = rec * 40
            ok = True
            cells = _np.zeros(81, _np.int8)
            h = _np.zeros(14, _np.int16)
            b0 = _np.uint32(buf[base]) | (_np.uint32(buf[base + 1]) << 8)
            k0 = (b0 >> 1) & 0x7F
            k1 = (b0 >> 8) & 0x7F
            cur = 15
            if k0 < 81:
                cells[sqmap[k0]] = 8
            if k1 < 81:
                cells[sqmap[k1]] = 22
            for sq in range(81):
                if sq == k0 or sq == k1:
                    continue
                byi = cur >> 3
                lo = _np.uint32(buf[base + byi])
                if byi + 1 < 32:
                    lo |= _np.uint32(buf[base + byi + 1]) << 8
                v = (lo >> (cur & 7)) & 0xFF
                pid = board_lut[v, 0]
                if pid < 0:
                    ok = False
                    break
                cells[sqmap[sq]] = pid
                cur += board_lut[v, 1]
            if ok:
                while cur < 256:
                    byi = cur >> 3
                    lo = _np.uint32(buf[base + byi])
                    if byi + 1 < 32:
                        lo |= _np.uint32(buf[base + byi + 1]) << 8
                    v = (lo >> (cur & 7)) & 0x7F
                    hidx = hand_lut[v, 0]
                    hcol = hand_lut[v, 1]
                    hb = hand_lut[v, 2]
                    if hb < 0:
                        ok = False
                        break
                    if hidx >= 0:
                        h[hcol * 7 + hidx] += 1
                    cur += hb
            if not ok:
                feat[rec, 21] = _np.nan
                plys[rec] = 0
                scores[rec] = 0
                continue
            s = _np.int32(buf[base + 32]) | (_np.int32(buf[base + 33]) << 8)
            if s >= 32768:
                s -= 65536
            scores[rec] = s
            plys[rec] = _np.int32(buf[base + 36]) | (_np.int32(buf[base + 37]) << 8)
            for k in range(7):
                feat[rec, k] = h[k] + h[7 + k]
            promoted = 0
            ksr = -1; ksc = -1; kwr = -1; kwc = -1
            for sq in range(81):
                pid = cells[sq]
                if pid == 0:
                    continue
                r = sq // 9
                c = sq % 9
                if pid == 8:
                    ksr = r; ksc = c
                elif pid == 22:
                    kwr = r; kwc = c
                else:
                    bse = (pid - 1) % 14 + 1
                    t = (bse - 1) if bse <= 7 else (7 + bse - 9)
                    if bse >= 9:
                        promoted += 1
                    if pid <= 14:
                        if r <= 2:
                            feat[rec, 7 + t] += 1
                    else:
                        if r >= 6:
                            feat[rec, 7 + t] += 1
            ring1 = 0
            for sq in range(81):
                pid = cells[sq]
                if pid == 0 or pid == 8 or pid == 22:
                    continue
                r = sq // 9
                c = sq % 9
                if pid <= 14:
                    if ksr >= 0 and max(abs(r - ksr), abs(c - ksc)) == 1:
                        ring1 += 1
                else:
                    if kwr >= 0 and max(abs(r - kwr), abs(c - kwc)) == 1:
                        ring1 += 1
            feat[rec, 20] = promoted
            feat[rec, 22] = 16 - ring1
            king = 0.0
            if ksr >= 0:
                raw_k = 7 - (ksr + 1)
                if raw_k >= 0:
                    sup = 0
                    for sq in range(81):
                        pid = cells[sq]
                        if pid == 0 or pid == 8 or pid > 14:
                            continue
                        r = sq // 9
                        c = sq % 9
                        if (max(abs(r - ksr), abs(c - ksc)) <= 2
                                and r <= ksr + 1):
                            sup += 1
                    if sup > 3:
                        sup = 3
                    king += (raw_k + 0.5) * (0.25 + 0.25 * sup)
            if kwr >= 0:
                raw_k = (kwr + 1) - 3
                if raw_k >= 0:
                    sup = 0
                    for sq in range(81):
                        pid = cells[sq]
                        if pid <= 14 or pid == 22:
                            continue
                        r = sq // 9
                        c = sq % 9
                        if (max(abs(r - kwr), abs(c - kwc)) <= 2
                                and r >= kwr - 1):
                            sup += 1
                    if sup > 3:
                        sup = 3
                    king += (raw_k + 0.5) * (0.25 + 0.25 * sup)
            feat[rec, 21] = king
        return feat, plys, scores

    def features_batch_file(path, cache=True, sample=None, chunk_records=2_000_000):
        cp = path + ".gpfeat.npz"
        if (sample is None and cache and os.path.exists(cp)
                and os.path.getmtime(cp) >= os.path.getmtime(path)):
            z = _np.load(cp)
            return z["feat"], z["ply"], z["score"]
        n = os.path.getsize(path) // PSV_RECORD_SIZE
        mm = _np.memmap(path, dtype=_np.uint8, mode="r")
        if sample is not None and 0 < sample < n:
            idx = _np.linspace(0, n - 1, int(sample)).astype(_np.int64)
            buf = _np.ascontiguousarray(
                mm[:n * PSV_RECORD_SIZE].reshape(n, PSV_RECORD_SIZE)[idx]
            ).reshape(-1)
            del mm
            return _nb_feat_all(buf, int(sample), _NB_BOARD, _NB_HAND, _NB_SQMAP)
        feat = _np.empty((n, 23), _np.float32)
        plys = _np.empty(n, _np.int32)
        scores = _np.empty(n, _np.int32)
        for st in range(0, n, chunk_records):
            en = min(n, st + chunk_records)
            buf = _np.array(mm[st * PSV_RECORD_SIZE:en * PSV_RECORD_SIZE])
            f, p, s = _nb_feat_all(buf, en - st, _NB_BOARD, _NB_HAND, _NB_SQMAP)
            feat[st:en] = f
            plys[st:en] = p
            scores[st:en] = s
            del buf, f, p, s
        del mm
        if cache and n <= 20_000_000:
            _np.savez_compressed(cp, feat=feat, ply=plys, score=scores)
        elif cache:
            print(f"[gp_core] {os.path.basename(path)}: n={n:,} — "
                  ".gpfeat.npz キャッシュ書込は省略 (巨大ファイル)", flush=True)
        return feat, plys, scores

    def gp_batch_file(path, ply_mode="auto"):
        buf = _np.fromfile(path, dtype=_np.uint8)
        n = buf.shape[0] // PSV_RECORD_SIZE
        v = GP_VIEW
        mode = PLY_MODES.index(ply_mode)
        return _nb_gp_all(
            buf, n, _NB_BOARD, _NB_HAND, _NB_SQMAP,
            _np.array(GP_HAND_VALS, _np.float64),
            _np.array(GP_CAMP_VALS, _np.float64),
            _np.array(_PLY_CALIB, _np.float64),
            v.game_phase_w_hand, v.game_phase_w_promo, v.game_phase_w_camp,
            v.game_phase_w_king, v.game_phase_hand_ref, v.game_phase_promo_ref,
            v.game_phase_camp_ref, v.game_phase_king_ref, v.game_phase_scale,
            v.game_phase_nyugyoku_king_onset, v.game_phase_nyugyoku_king_full,
            v.game_phase_w_ply, v.game_phase_ply_ref,
            v.game_phase_w_exposure, v.game_phase_exposure_ref, mode)


PLY_MODES = ("record", "zero", "state", "auto")

_HAND_CAPS = (18, 4, 4, 4, 4, 2, 2)
_MATE_CP_NOISE = 30000


def noise_mask_np(feat, ply, score):
    import numpy as _np2
    n = feat.shape[0]
    ok = ~_np2.isnan(feat[:, 21])
    reasons = {"decode": int(n - ok.sum())}
    m = _np2.ones(n, bool)
    for k in range(7):
        m &= feat[:, k] <= _HAND_CAPS[k]
    reasons["hand_cap"] = int((ok & ~m).sum())
    ok &= m
    m = (feat[:, 20] <= 26) & (feat[:, 22] >= 0) & (feat[:, 22] <= 16) & (feat[:, 21] <= 13.0)
    reasons["feat_range"] = int((ok & ~m).sum())
    ok &= m
    m = (ply >= 1) & (ply <= 512)
    reasons["ply_range"] = int((ok & ~m).sum())
    ok &= m
    m = _np2.abs(score) < 32767
    reasons["score_sentinel"] = int((ok & ~m).sum())
    ok &= m
    reasons["dropped"] = int(n - ok.sum())
    return ok, reasons


def expand_bin_paths(path):
    if os.path.isdir(path):
        import glob as _g
        return sorted(_g.glob(os.path.join(path, "*.bin")))
    return [path]


def expand_bin_paths_done(path, require_done=False):
    if os.path.isdir(path):
        import glob as _g
        all_bins = sorted(_g.glob(os.path.join(path, "*.bin")))
        if require_done:
            ready = [b for b in all_bins if os.path.exists(b + ".done")]
            skipped = len(all_bins) - len(ready)
            if skipped:
                print(f"[gp] {skipped} bin スキップ (.done なし / 処理中)")
            return ready
        return all_bins
    return [path]


_DEFAULT_SPLIT_BYTES = 19_558_599_240


class ChunkedBinWriter:

    def __init__(self, path_template, max_bytes=_DEFAULT_SPLIT_BYTES):
        self._base = path_template[:-4] if path_template.endswith(".bin") else path_template
        self._max = max_bytes
        self._chunk = 0
        self._written = 0
        self._f = None
        self._files = []
        if max_bytes <= 0:
            self._f = open(path_template, "wb")
            self._files.append(path_template)
            self._max = 0

    def _open_next(self):
        if self._f is not None:
            self._f.close()
        self._chunk += 1
        p = f"{self._base}_{self._chunk:03d}.bin"
        self._f = open(p, "wb")
        self._files.append(p)
        self._written = 0

    def write(self, data):
        if not data:
            return
        if self._max <= 0:
            self._f.write(data)
            return
        if self._f is None or self._written >= self._max:
            self._open_next()
        self._f.write(data)
        self._written += len(data)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def files(self):
        return self._files


def gp_of_features(feats, ply: float, hand_vals=None, camp_vals=None) -> float:
    h7, c13, promo, king, expo = feats
    hv = hand_vals if hand_vals is not None else GP_HAND_VALS
    cv = camp_vals if camp_vals is not None else GP_CAMP_VALS
    whand = sum(h7[k] * hv[k] for k in range(7))
    wcamp = sum(c13[k] * cv[k] for k in range(13))
    return _clamp(_calc_state_phase(whand, promo, wcamp, king, expo,
                                    cfg=GP_VIEW, ply=ply), 0.0, 1.5)


def gp_of_record_v2(raw40: bytes, ply_mode: str = "auto"):
    data32, score, _mv, gameply, _res, _pad = struct.unpack(PSV_FMT, raw40)
    sb = SvBoard()
    _fast_decode_psfen_into(data32, sb)
    feats = features_from_cells(sb)
    if ply_mode == "record":
        ply = float(gameply)
    elif ply_mode == "zero":
        ply = 0.0
    elif ply_mode == "state":
        ply = estimate_ply_from_state(gp_of_features(feats, 0.0))
    else:
        ply = (float(gameply) if gameply >= 1
               else estimate_ply_from_state(gp_of_features(feats, 0.0)))
    return gp_of_features(feats, ply), score, gameply



GP_VIEW = _GpParamView(_GAME_PHASE_DEFAULT_VEC)

def gp_of_sfen(sfen_body: str, ply: int = 0) -> float:
    b = MiniBoard(sfen_body)
    whand, promo, camp, king, expo = _game_phase_state_counts_core(b)
    return _clamp(_calc_state_phase(whand, promo, camp, king, expo,
                                    cfg=GP_VIEW, ply=ply), 0.0, 1.5)


def gp_of_record(raw40: bytes):
    data32, score, _mv, gameply, _res, _pad = struct.unpack(PSV_FMT, raw40)
    sb = SvBoard()
    _fast_decode_psfen_into(data32, sb)
    return gp_of_sfen(_svboard_to_sfen(sb), ply=gameply), score, gameply


def iter_psv(path):
    with open(path, "rb") as f:
        while True:
            raw = f.read(PSV_RECORD_SIZE)
            if len(raw) < PSV_RECORD_SIZE:
                return
            yield raw



_KIF_ZEN = {c: i + 1 for i, c in enumerate("１２３４５６７８９")}
_KIF_KAN = {c: i + 1 for i, c in enumerate("一二三四五六七八九")}
_KIF_PIECE = {"歩": (1, 0), "香": (2, 1), "桂": (3, 2), "銀": (4, 3),
              "金": (5, 4), "角": (6, 5), "飛": (7, 6), "玉": (8, None),
              "王": (8, None), "と": (9, None), "成香": (10, None),
              "成桂": (11, None), "成銀": (12, None), "馬": (13, None),
              "竜": (14, None), "龍": (14, None)}
_KIF_PROMOTE = {1: 9, 2: 10, 3: 11, 4: 12, 6: 13, 7: 14}
_KIF_DEMOTE = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
               9: 0, 10: 1, 11: 2, 12: 3, 13: 5, 14: 6}
import re as _re
_KIF_MOVE_RE = _re.compile(
    r"^\s*(\d+)\s+(同|[１-９][一二三四五六七八九])[　\s]*"
    r"(成香|成桂|成銀|と|馬|竜|龍|歩|香|桂|銀|金|角|飛|玉|王)"
    r"(成?)(打?)(?:\((\d)(\d)\))?")
_PID_OF_SFEN = {p: i for i, p in enumerate(_PID_SFEN) if p}
_HIRATE_BOARD = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"


def _kif_hirate():
    sb = SvBoard()
    r = 0
    for row in _HIRATE_BOARD.split("/"):
        c = 0
        for ch in row:
            if ch.isdigit():
                c += int(ch)
            else:
                sb.cells[r * 9 + c] = _PID_OF_SFEN[ch]
                c += 1
        r += 1
    return sb


def iter_kif_features(path):
    raw = open(path, "rb").read()
    text = None
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return
    sb = _kif_hirate()
    prev = None
    for ln in text.replace("\r\n", "\n").split("\n"):
        if ln.lstrip().startswith("変化"):
            return
        m = _KIF_MOVE_RE.match(ln)
        if not m:
            continue
        no = int(m.group(1))
        d = m.group(2)
        dest = prev if d == "同" else (_KIF_ZEN[d[0]], _KIF_KAN[d[1]])
        if dest is None:
            return
        side = 0 if no % 2 == 1 else 1
        off = side * 14
        f9, r9 = dest
        dsq = (r9 - 1) * 9 + (9 - f9)
        cap = sb.cells[dsq]
        if cap:
            base = (cap - 1) % 14 + 1
            idx = _KIF_DEMOTE.get(base)
            if idx is None:
                return
            sb.hand[side][idx] += 1
        org = (int(m.group(6)), int(m.group(7))) if m.group(6) else None
        if m.group(5) == "打" or org is None:
            pid, hidx = _KIF_PIECE[m.group(3)]
            if sb.hand[side][hidx] > 0:
                sb.hand[side][hidx] -= 1
            sb.cells[dsq] = pid + off
        else:
            osq = (org[1] - 1) * 9 + (9 - org[0])
            pid = sb.cells[osq]
            if pid == 0:
                return
            sb.cells[osq] = 0
            if m.group(4) == "成":
                base = (pid - 1) % 14 + 1
                pro = _KIF_PROMOTE.get(base)
                if pro is not None:
                    pid = pro + (14 if pid > 14 else 0)
            sb.cells[dsq] = pid
        prev = dest
        yield no, features_from_cells(sb)


def gp_fit(kif_dir=None, bin_path=None, bin_sample=30000, iters=2000,
           pairs_per_game=400, bin_pairs=20000, min_dply_kif=8,
           min_dply_bin=40, lam=0.005, seed=7, out_json="gp_fit_result.json",
           wr_pairs=50000, wr_weight=0.25, wr_dply=10, wr_dcp=800):
    import glob as _glob
    import json as _json
    import numpy as np
    rng = np.random.default_rng(seed)
    feats = []
    gid = 0
    n_games = 0
    if kif_dir:
        for p in sorted(_glob.glob(os.path.join(kif_dir, "*.kif"))):
            got = 0
            for no, f3 in iter_kif_features(p):
                feats.append((gid, no) + f3)
                got += 1
            if got:
                gid += 1
                n_games += 1
    n_kif = len(feats)
    bin_scores = []
    if bin_path:
        _files = expand_bin_paths(bin_path)
        if not _files:
            raise SystemExit(f"[gp-fit] .bin が見つかりません: {bin_path}")
        _budget = max(1, bin_sample // len(_files))
        _dropped = 0
        for _bf in _files:
            total = os.path.getsize(_bf) // PSV_RECORD_SIZE
            stride = max(1, total // _budget)
            with open(_bf, "rb") as f:
                for k in range(min(_budget, total)):
                    f.seek(k * stride * PSV_RECORD_SIZE)
                    raw = f.read(PSV_RECORD_SIZE)
                    if len(raw) < PSV_RECORD_SIZE:
                        break
                    data32, _s, _m, ply, _r, _p = struct.unpack(PSV_FMT, raw)
                    try:
                        sb = SvBoard()
                        _fast_decode_psfen_into(data32, sb)
                        f3 = features_from_cells(sb)
                    except Exception:
                        _dropped += 1
                        continue
                    h7 = f3[0]
                    if (any(h7[i] > _HAND_CAPS[i] for i in range(7))
                            or f3[2] > 26 or not (0 <= f3[4] <= 16)
                            or f3[3] > 13.0 or not (1 <= ply <= 512)
                            or abs(int(_s)) >= 32767):
                        _dropped += 1
                        continue
                    feats.append((-1, int(ply)) + f3)
                    bin_scores.append(int(_s))
        if _dropped:
            print(f"[gp-fit][NOISE] {_dropped} 局面を除外 "
                  f"(decode不良/特徴範囲外/ply異常/scoreセンチネル)")
    n = len(feats)
    if n < 100:
        raise SystemExit(f"[gp-fit] 局面不足 (n={n}) — --kif-dir / --bin を確認")
    G = np.array([f[0] for f in feats])
    PLY = np.array([f[1] for f in feats], np.float64)
    H = np.array([f[2] for f in feats], np.float64)
    C = np.array([f[3] for f in feats], np.float64)
    PR = np.array([f[4] for f in feats], np.float64)
    KG = np.array([f[5] for f in feats], np.float64)
    EX = np.array([f[6] for f in feats], np.float64)

    ei, lj = [], []
    for g in range(gid):
        idx = np.where(G == g)[0]
        if len(idx) < 2:
            continue
        a = rng.integers(0, len(idx), pairs_per_game)
        b = rng.integers(0, len(idx), pairs_per_game)
        pa, pb = idx[np.minimum(a, b)], idx[np.maximum(a, b)]
        ok = PLY[pb] - PLY[pa] >= min_dply_kif
        ei.extend(pa[ok]); lj.extend(pb[ok])
    bidx = np.where(G == -1)[0]
    if len(bidx) >= 2 and bin_pairs > 0:
        a = bidx[rng.integers(0, len(bidx), bin_pairs * 3)]
        b = bidx[rng.integers(0, len(bidx), bin_pairs * 3)]
        sw = PLY[a] > PLY[b]
        a2 = np.where(sw, b, a); b2 = np.where(sw, a, b)
        ok = PLY[b2] - PLY[a2] >= min_dply_bin
        ei.extend(a2[ok][:bin_pairs]); lj.extend(b2[ok][:bin_pairs])
    EI = np.array(ei); LJ = np.array(lj)
    npair = len(EI)
    if npair < 100:
        raise SystemExit(f"[gp-fit] ペア不足 ({npair})")
    margin = np.minimum(0.02, 0.0004 * (PLY[LJ] - PLY[EI]))

    WLO = WHI = None
    if wr_pairs > 0 and len(bin_scores) >= 2:
        babs = np.abs(np.array(bin_scores, np.float64))
        bply = PLY[G == -1]
        bidx2 = np.where(G == -1)[0]
        a = rng.integers(0, len(bidx2), wr_pairs * 4)
        b = rng.integers(0, len(bidx2), wr_pairs * 4)
        ok = (np.abs(bply[a] - bply[b]) <= wr_dply) & \
             (np.abs(babs[a] - babs[b]) >= wr_dcp) & \
             (babs[a] < _MATE_CP_NOISE) & (babs[b] < _MATE_CP_NOISE)
        a, b = a[ok][:wr_pairs], b[ok][:wr_pairs]
        sw = babs[a] > babs[b]
        lo = np.where(sw, b, a); hi = np.where(sw, a, b)
        WLO, WHI = bidx2[lo], bidx2[hi]
        print(f"[gp-fit] WR-PAIR: {len(WLO)} ペア (|Δply|≤{wr_dply}, "
              f"Δ|cp|≥{wr_dcp}, weight={wr_weight})")

    v = GP_VIEW
    def gp_all(hv, cv):
        S = (v.game_phase_w_hand * (H @ hv) / v.game_phase_hand_ref
             + v.game_phase_w_promo * PR / v.game_phase_promo_ref
             + v.game_phase_w_camp * (C @ cv) / v.game_phase_camp_ref
             + v.game_phase_w_king * KG / v.game_phase_king_ref
             + v.game_phase_w_ply * PLY / v.game_phase_ply_ref
             + v.game_phase_w_exposure * EX / v.game_phase_exposure_ref)
        core = 1.0 - np.exp(np.maximum(-S / v.game_phase_scale, -50.0))
        ramp = (KG - v.game_phase_nyugyoku_king_onset) / (
            v.game_phase_nyugyoku_king_full - v.game_phase_nyugyoku_king_onset)
        gp = core + 0.5 * np.clip(ramp, 0.0, 1.0)
        return np.clip(gp, 0.0, 1.5)

    th0 = np.array(GP_HAND_VALS[1:] + list(GP_CAMP_VALS), np.float64)
    LO = np.full(19, 0.0); LO[:6] = 0.1
    HI = np.full(19, 25.0)
    def unpack(th):
        return (np.concatenate(([1.0], th[:6])), th[6:])
    def loss(th):
        hv, cv = unpack(th)
        gp = gp_all(hv, cv)
        viol = np.maximum(gp[EI] - gp[LJ] + margin, 0.0)
        total = viol.mean() + lam * ((th - th0) ** 2).mean()
        if WLO is not None and len(WLO):
            wviol = np.maximum(gp[WLO] - gp[WHI] + 0.005, 0.0)
            total += wr_weight * wviol.mean()
        return total
    def viol_rate(th):
        hv, cv = unpack(th)
        gp = gp_all(hv, cv)
        return float((gp[EI] >= gp[LJ]).mean())

    l0, vr0 = loss(th0), viol_rate(th0)
    th = th0.copy()
    c0, A, alpha, gamma = 0.15, iters * 0.1, 0.602, 0.101
    target_step = 0.08
    g_est = 0.0
    for _ in range(8):
        delta = rng.choice((-1.0, 1.0), 19)
        g_est += abs(loss(np.clip(th + c0 * delta, LO, HI))
                     - loss(np.clip(th - c0 * delta, LO, HI))) / (2 * c0)
    g_est = max(g_est / 8.0, 1e-9)
    a0 = target_step / g_est * (1 + A) ** alpha
    for k in range(iters):
        ck = c0 / (k + 1) ** gamma
        ak = a0 / (k + 1 + A) ** alpha
        delta = rng.choice((-1.0, 1.0), 19)
        lp = loss(np.clip(th + ck * delta, LO, HI))
        lm = loss(np.clip(th - ck * delta, LO, HI))
        th = np.clip(th - ak * (lp - lm) / (2 * ck) * delta, LO, HI)
        if (k + 1) % max(1, iters // 10) == 0:
            print(f"[gp-fit] iter {k+1}/{iters} loss={loss(th):.6f}")
    l1, vr1 = loss(th), viol_rate(th)
    hv1, cv1 = unpack(th)
    res = {
        "hand_vals": [round(float(x), 4) for x in hv1],
        "camp_vals": [round(float(x), 4) for x in cv1],
        "loss": {"init": float(l0), "final": float(l1)},
        "pair_violation_rate": {"init": vr0, "final": vr1},
        "data": {"kif_games": n_games, "kif_positions": n_kif,
                 "bin_positions": n - n_kif, "pairs": npair},
        "frozen": "15-vec (W_*/Ref_*/Scale/Onset/Full) は凍結。hv[P]=1 アンカー",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        _json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[gp-fit] loss {l0:.6f}→{l1:.6f} / ペア違反率 {vr0:.4f}→{vr1:.4f}")
    print(f"[gp-fit] hand_vals (P固定1): "
          + " ".join(f"{nm}={x:.2f}" for nm, x in zip("PLNSGBR", hv1)))
    print(f"[gp-fit] camp_vals: "
          + " ".join(f"{nm}={x:.2f}" for nm, x in zip(CAMP_TYPE_NAMES, cv1)))
    print(f"[gp-fit] → {out_json} (適用は GP_HAND_VALS/GP_CAMP_VALS と "
          "Pyfamate の GP_Hand_Val_*/GP_Camp_Val_* を同値更新)")
    return res



try:
    import torch as _torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False



def _torch_pick_device(arg="auto"):
    if not _TORCH_OK:
        sys.exit("[gp-torch] torch 未導入。`pip install torch` を使用してください")
    if arg == "cpu":
        return _torch.device("cpu")
    if arg == "cuda" or (arg == "auto" and _torch.cuda.is_available()):
        if not _torch.cuda.is_available():
            sys.exit("[gp-torch] --device cuda 指定ですが CUDA が利用できません")
        return _torch.device("cuda")
    return _torch.device("cpu")


def _torch_vec_params():
    v = GP_VIEW
    return dict(
        wh=v.game_phase_w_hand, wpm=v.game_phase_w_promo, wc=v.game_phase_w_camp,
        wk=v.game_phase_w_king, hr=v.game_phase_hand_ref, pr=v.game_phase_promo_ref,
        cr=v.game_phase_camp_ref, kr=v.game_phase_king_ref, scale=v.game_phase_scale,
        onset=v.game_phase_nyugyoku_king_onset, full=v.game_phase_nyugyoku_king_full,
        wply=v.game_phase_w_ply, plyref=v.game_phase_ply_ref,
        wexpo=v.game_phase_w_exposure, exporef=v.game_phase_exposure_ref)


def gp_tensor(feat, ply, hv, cv, vp):
    H = feat[:, 0:7]
    C = feat[:, 7:20]
    PR = feat[:, 20]
    KG = feat[:, 21]
    EX = feat[:, 22]
    S = (vp["wh"] * (H @ hv) / vp["hr"]
         + vp["wpm"] * PR / vp["pr"]
         + vp["wc"] * (C @ cv) / vp["cr"]
         + vp["wk"] * KG / vp["kr"]
         + vp["wply"] * ply / vp["plyref"]
         + vp["wexpo"] * EX / vp["exporef"])
    core = 1.0 - _torch.exp(_torch.clamp(-S / vp["scale"], min=-50.0))
    ramp = _torch.clamp((KG - vp["onset"]) / (vp["full"] - vp["onset"]), 0.0, 1.0)
    return _torch.clamp(core + 0.5 * ramp, 0.0, 1.5)


def _torch_load_dataset(kif_dir, bin_path, bin_sample, device):
    import numpy as np
    rows, plys, gids = [], [], []
    gid = 0
    n_games = 0
    if kif_dir:
        import glob
        for p in sorted(glob.glob(os.path.join(kif_dir, "*.kif"))):
            got = 0
            for no, f3 in iter_kif_features(p):
                h7, c13, pr, ki, ex = f3
                rows.append(list(h7) + list(c13) + [pr, ki, ex])
                plys.append(no)
                gids.append(gid)
                got += 1
            if got:
                gid += 1
                n_games += 1
    n_kif = len(rows)
    if bin_path:
        _files = expand_bin_paths(bin_path)
        if not _files:
            sys.exit(f"[gp-torch] .bin が見つかりません: {bin_path}")
        _sizes = [max(1, os.path.getsize(_bf) // PSV_RECORD_SIZE)
                  for _bf in _files]
        _total = sum(_sizes)
        _fs, _ps, _ss = [], [], []
        for _bf, _sz in zip(_files, _sizes):
            _q = None
            if bin_sample and _total > bin_sample:
                _q = max(1, int(round(bin_sample * (_sz / _total))))
            _f, _p, _s = features_batch_file(_bf, sample=_q)
            _fs.append(_f); _ps.append(_p); _ss.append(_s)
            del _f, _p, _s
        F = np.concatenate(_fs); P = np.concatenate(_ps); S = np.concatenate(_ss)
        del _fs, _ps, _ss
        n = F.shape[0]
        if bin_sample and bin_sample < n:
            idx = np.linspace(0, n - 1, bin_sample).astype(np.int64)
            F, P, S = F[idx], P[idx], S[idx]
        good, _reasons = noise_mask_np(F, P, S)
        if _reasons["dropped"]:
            print(f"[gp-torch][NOISE] {_reasons['dropped']}/{len(good)} 局面を除外 "
                  + " ".join(f"{k}={v}" for k, v in _reasons.items()
                             if k != "dropped" and v))
        F, P, S = F[good], P[good], S[good]
        rows_np = np.asarray(rows, np.float32) if rows else np.zeros((0, 23), np.float32)
        feat = np.concatenate([rows_np, F.astype(np.float32)])
        ply = np.concatenate([np.asarray(plys, np.float32),
                              P.astype(np.float32)])
        gida = np.concatenate([np.asarray(gids, np.int64),
                               np.full(F.shape[0], -1, np.int64)])
        sca = np.concatenate([np.zeros(n_kif, np.float64),
                              S.astype(np.float64)])
    else:
        import numpy as np
        feat = np.asarray(rows, np.float32)
        ply = np.asarray(plys, np.float32)
        gida = np.asarray(gids, np.int64)
        sca = np.zeros(n_kif, np.float64)
    t = lambda a, dt: _torch.tensor(a, dtype=dt, device=device)
    return (t(feat, _torch.float64), t(ply, _torch.float64), gida,
            n_games, n_kif, sca)


def _torch_build_pairs(ply_np, gid_np, pairs_per_game, bin_pairs,
                       min_dply_kif, min_dply_bin, rng):
    import numpy as np
    ei, lj = [], []
    for g in range(gid_np.max() + 1 if len(gid_np) and gid_np.max() >= 0 else 0):
        idx = np.where(gid_np == g)[0]
        if len(idx) < 2:
            continue
        a = rng.integers(0, len(idx), pairs_per_game)
        b = rng.integers(0, len(idx), pairs_per_game)
        pa, pb = idx[np.minimum(a, b)], idx[np.maximum(a, b)]
        ok = ply_np[pb] - ply_np[pa] >= min_dply_kif
        ei.extend(pa[ok]); lj.extend(pb[ok])
    bidx = np.where(gid_np == -1)[0]
    if len(bidx) >= 2 and bin_pairs > 0:
        a = bidx[rng.integers(0, len(bidx), bin_pairs * 3)]
        b = bidx[rng.integers(0, len(bidx), bin_pairs * 3)]
        sw = ply_np[a] > ply_np[b]
        a2 = np.where(sw, b, a); b2 = np.where(sw, a, b)
        ok = ply_np[b2] - ply_np[a2] >= min_dply_bin
        ei.extend(a2[ok][:bin_pairs]); lj.extend(b2[ok][:bin_pairs])
    return np.asarray(ei, np.int64), np.asarray(lj, np.int64)


def _torch_cmd_fit(a):
    import json
    import numpy as np
    import time
    dev = _torch_pick_device(a.device)
    print(f"[gp-torch] device={dev}"
          + (f" ({_torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    feat, ply, gid_np, n_games, n_kif, sc_np = _torch_load_dataset(
        a.kif_dir, a.bin, a.bin_sample, dev)
    n = feat.shape[0]
    if n < 100:
        sys.exit(f"[gp-torch] 局面不足 (n={n})")
    rng = np.random.default_rng(a.seed)
    ply_np = ply.cpu().numpy()
    EI, LJ = _torch_build_pairs(ply_np, gid_np, a.pairs_per_game, a.bin_pairs,
                                a.min_dply_kif, a.min_dply_bin, rng)
    if len(EI) < 100:
        sys.exit(f"[gp-torch] ペア不足 ({len(EI)})")
    EI_t = _torch.tensor(EI, device=dev)
    LJ_t = _torch.tensor(LJ, device=dev)
    margin = _torch.tensor(
        np.minimum(0.02, 0.0004 * (ply_np[LJ] - ply_np[EI])),
        dtype=_torch.float64, device=dev)
    WLO_t = WHI_t = None
    if a.wr_pairs > 0:
        babs = np.abs(sc_np)
        bidx2 = np.where(gid_np == -1)[0]
        if len(bidx2) >= 2:
            aa = bidx2[rng.integers(0, len(bidx2), a.wr_pairs * 4)]
            bb = bidx2[rng.integers(0, len(bidx2), a.wr_pairs * 4)]
            ok = (np.abs(ply_np[aa] - ply_np[bb]) <= a.wr_dply) & \
                 (np.abs(babs[aa] - babs[bb]) >= a.wr_dcp) & \
                 (babs[aa] < _MATE_CP_NOISE) & (babs[bb] < _MATE_CP_NOISE)
            aa, bb = aa[ok][:a.wr_pairs], bb[ok][:a.wr_pairs]
            sw = babs[aa] > babs[bb]
            lo = np.where(sw, bb, aa); hi = np.where(sw, aa, bb)
            if len(lo):
                WLO_t = _torch.tensor(lo, device=dev)
                WHI_t = _torch.tensor(hi, device=dev)
                print(f"[gp-torch] WR-PAIR: {len(lo)} ペア "
                      f"(|Δply|≤{a.wr_dply}, Δ|cp|≥{a.wr_dcp}, "
                      f"weight={a.wr_weight})")

    hv0 = _torch.tensor(GP_HAND_VALS, dtype=_torch.float64, device=dev)
    cv0 = _torch.tensor(GP_CAMP_VALS, dtype=_torch.float64, device=dev)
    hv_free = hv0[1:].clone().requires_grad_(True)
    cv = cv0.clone().requires_grad_(True)
    params = [hv_free, cv]
    vp0 = _torch_vec_params()
    vp = dict(vp0)
    if a.learn == "full":
        vt = {k: _torch.tensor(float(v), dtype=_torch.float64, device=dev,
                                requires_grad=True)
              for k, v in vp0.items() if k not in ("full",)}
        d_full = _torch.tensor(float(vp0["full"] - vp0["onset"]),
                               dtype=_torch.float64, device=dev).log()
        d_full.requires_grad_(True)
        params += list(vt.values()) + [d_full]
        def vp_now():
            v = dict(vt)
            v["full"] = vt["onset"] + _torch.nn.functional.softplus(d_full)
            return v
    else:
        def vp_now():
            return vp

    opt = _torch.optim.Adam(params, lr=a.lr)
    def gp_all():
        hv = _torch.cat([hv0[:1], hv_free])
        return gp_tensor(feat, ply, hv, cv, vp_now())
    def loss_fn():
        gp = gp_all()
        viol = _torch.relu(gp[EI_t] - gp[LJ_t] + margin)
        reg = a.lam * (((hv_free - hv0[1:]) ** 2).mean()
                       + ((cv - cv0) ** 2).mean())
        total = viol.mean() + reg
        if WLO_t is not None:
            total = total + a.wr_weight * _torch.relu(
                gp[WLO_t] - gp[WHI_t] + 0.005).mean()
        return total, gp
    with _torch.no_grad():
        l0, gp = loss_fn()
        vr0 = float((gp[EI_t] >= gp[LJ_t]).double().mean())
    t0 = time.perf_counter()
    for k in range(a.iters):
        opt.zero_grad()
        l, _ = loss_fn()
        l.backward()
        opt.step()
        with _torch.no_grad():
            hv_free.clamp_(0.1, 25.0)
            cv.clamp_(0.0, 25.0)
        if (k + 1) % max(1, a.iters // 10) == 0:
            print(f"[gp-torch] iter {k+1}/{a.iters} loss={float(l):.6f}")
    el = time.perf_counter() - t0
    with _torch.no_grad():
        l1, gp = loss_fn()
        vr1 = float((gp[EI_t] >= gp[LJ_t]).double().mean())
        hv1 = _torch.cat([hv0[:1], hv_free]).cpu().numpy()
        cv1 = cv.cpu().numpy()
    res = {
        "hand_vals": [round(float(x), 4) for x in hv1],
        "camp_vals": [round(float(x), 4) for x in cv1],
        "loss": {"init": float(l0), "final": float(l1)},
        "pair_violation_rate": {"init": vr0, "final": vr1},
        "data": {"kif_games": n_games, "kif_positions": n_kif,
                 "bin_positions": int(n - n_kif), "pairs": int(len(EI))},
        "engine": {"device": str(dev), "optimizer": "adam", "lr": a.lr,
                   "iters": a.iters, "seconds": round(el, 1),
                   "learn": a.learn},
    }
    if a.learn == "full":
        res["vec"] = {k: round(float(v), 6) for k, v in vp_now().items()}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[gp-torch] loss {float(l0):.6f}→{float(l1):.6f} / "
          f"ペア違反率 {vr0:.4f}→{vr1:.4f} ({el:.1f}s, {a.iters} iters)")
    print("[gp-torch] hand:", " ".join(
        f"{nm}={x:.2f}" for nm, x in zip("PLNSGBR", hv1)))
    print("[gp-torch] camp:", " ".join(
        f"{nm}={x:.2f}" for nm, x in zip(CAMP_TYPE_NAMES, cv1)))
    print(f"[gp-torch] → {a.out}")


def _torch_apply_tables(path):
    if not path:
        return
    import json
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    GP_HAND_VALS[:] = r["hand_vals"]
    GP_CAMP_VALS[:] = r["camp_vals"]
    print(f"[gp-torch] tables loaded: {path}")


_DEFAULT_GROUP_EDGES = ",".join(f"{x/10:.1f}" for x in range(16))


def _torch_group_one(bin_path, dev, edges, ply_mode, out_dir, report_only, writers=None):
    import numpy as np
    F, P, S = features_batch_file(bin_path)
    n = F.shape[0]
    feat = _torch.tensor(F, dtype=_torch.float64, device=dev)
    ply_rec = _torch.tensor(P, dtype=_torch.float64, device=dev)
    hv = _torch.tensor(GP_HAND_VALS, dtype=_torch.float64, device=dev)
    cv = _torch.tensor(GP_CAMP_VALS, dtype=_torch.float64, device=dev)
    vp = _torch_vec_params()
    if ply_mode == "record":
        ply = ply_rec
    elif ply_mode == "zero":
        ply = _torch.zeros_like(ply_rec)
    else:
        gp0 = gp_tensor(feat, _torch.zeros_like(ply_rec), hv, cv, vp)
        calib = np.asarray(_PLY_CALIB, np.float64)
        est = np.interp(gp0.cpu().numpy() / 0.05,
                        np.arange(len(calib)), calib)
        est_t = _torch.tensor(est, dtype=_torch.float64, device=dev)
        ply = est_t if ply_mode == "state" else _torch.where(
            ply_rec >= 1, ply_rec, est_t)
    gp = gp_tensor(feat, ply, hv, cv, vp)
    gp_np = gp.cpu().numpy()
    del feat, ply_rec, ply, gp
    nb = len(edges) - 1
    bad = np.isnan(gp_np)
    ids = np.clip(np.searchsorted(np.asarray(edges), gp_np, side="right") - 1,
                  0, nb - 1)
    if not report_only and writers is not None:
        recs = np.fromfile(bin_path, dtype=np.uint8)[:n * 40].reshape(n, 40)
        for i in range(nb):
            sel = (ids == i) & ~bad
            if sel.any():
                writers[i].write(recs[sel].tobytes())
        del recs
    good = ~bad
    counts = [0] * nb
    ssum = [0.0] * nb
    psum = [0.0] * nb
    for i in range(nb):
        sel = (ids == i) & good
        c = int(sel.sum())
        counts[i] = c
        if c:
            ssum[i] = float(np.abs(S[sel]).sum())
            psum[i] = float(P[sel].sum())
    return {"counts": counts, "ssum": ssum, "psum": psum,
            "n": n, "bad": int(bad.sum())}


def _torch_cmd_group(a):
    import numpy as np
    import time
    dev = _torch_pick_device(a.device)
    _torch_apply_tables(a.tables)
    edges = sorted(float(x) for x in a.edges.split(","))
    nb = len(edges) - 1

    if os.path.isdir(a.bin):
        bin_files = expand_bin_paths_done(a.bin, require_done=a.require_done)
        if not bin_files:
            sys.exit(f"[gp-torch] .done 付き .bin が見つかりません: {a.bin}")
        stem = os.path.basename(os.path.abspath(a.bin).rstrip("/\\")) or "group"
    else:
        bin_files = [a.bin]
        stem = os.path.splitext(os.path.basename(a.bin))[0]

    out_dir = a.out_dir or os.path.join(".", f"{stem}_gp")
    os.makedirs(out_dir, exist_ok=True)

    writers = {}
    if not a.report_only:
        for i in range(nb):
            writers[i] = ChunkedBinWriter(os.path.join(
                out_dir, f"{stem}_gp_{edges[i]:.2f}-{edges[i+1]:.2f}.bin"),
                max_bytes=a.split_bytes)

    t0 = time.perf_counter()
    total_counts = [0] * nb
    total_ssum = [0.0] * nb
    total_psum = [0.0] * nb
    total_n = 0
    total_bad = 0
    for fi, bf in enumerate(bin_files):
        print(f"[gp-torch] ({fi+1}/{len(bin_files)}) {os.path.basename(bf)} ...", flush=True)
        st = _torch_group_one(bf, dev, edges, a.ply_mode, out_dir,
                              a.report_only, writers if writers else None)
        for i in range(nb):
            total_counts[i] += st["counts"][i]
            total_ssum[i] += st["ssum"][i]
            total_psum[i] += st["psum"][i]
        total_n += st["n"]
        total_bad += st["bad"]
    for w in writers.values():
        w.close()
    el = time.perf_counter() - t0

    lines = [f"# GP grouping report — {a.bin} ({len(bin_files)} files) [gp_torch/{dev}]",
             f"# ply_mode={a.ply_mode} edges={edges} 処理={total_n} 不良={total_bad} "
             f"({el:.2f}s, {total_n/max(el,1e-9):,.0f} pos/s)", ""]
    lines.append(f"{'bucket':22s} {'count':>10s} {'%':>7s} {'avg|cp|':>9s} {'avg ply':>8s}")
    total_good = total_n - total_bad
    for i in range(nb):
        c = total_counts[i]
        pct = 100.0 * c / max(1, total_good)
        asc = total_ssum[i] / c if c else 0.0
        apl = total_psum[i] / c if c else 0.0
        lines.append(f"GP[{edges[i]:.2f},{edges[i+1]:.2f}) {c:>10d} {pct:>6.2f}% "
                     f"{asc:>9.1f} {apl:>8.1f}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, f"{stem}_gp_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n")
    if not a.report_only:
        print(f"[gp-torch] buckets → {out_dir}/{stem}_gp_*.bin")


def _torch_cmd_verify(a):
    import numpy as np
    dev = _torch_pick_device(a.device)
    F, P, _ = features_batch_file(a.bin)
    n = min(a.n, F.shape[0])
    idx = np.linspace(0, F.shape[0] - 1, n).astype(np.int64)
    feat = _torch.tensor(F[idx], dtype=_torch.float64, device=dev)
    ply = _torch.tensor(P[idx], dtype=_torch.float64, device=dev)
    hv = _torch.tensor(GP_HAND_VALS, dtype=_torch.float64, device=dev)
    cv = _torch.tensor(GP_CAMP_VALS, dtype=_torch.float64, device=dev)
    gp_t = gp_tensor(feat, ply, hv, cv, _torch_vec_params()).cpu().numpy()
    mx = 0.0
    with open(a.bin, "rb") as f:
        for k, i in enumerate(idx):
            f.seek(int(i) * PSV_RECORD_SIZE)
            raw = f.read(PSV_RECORD_SIZE)
            g_ref, _s, _p = gp_of_record_v2(raw, "record")
            mx = max(mx, abs(g_ref - gp_t[k]))
    print(f"[gp-torch] verify n={n} device={dev} max|Δgp|={mx:.2e}")
    sys.exit(0 if mx < 1e-9 else 1)


def _cli():
    import argparse
    if len(sys.argv) > 1 and sys.argv[1].startswith("torch-"):
        _cli_torch()
        return
    ap = argparse.ArgumentParser(description="Pyfamate GP 計算部 (独立版)")
    ap.add_argument("--sfen", help="SFEN 文字列の GP を表示")
    ap.add_argument("--ply", type=int, default=0)
    ap.add_argument("--verify", metavar="BIN",
                    help="PSV bin で Pyfamate 本体 import 経路と等価性検証")
    ap.add_argument("--verify-fast", metavar="BIN",
                    help="features_from_cells(高速路) と MiniBoard 経路の全量一致検証")
    ap.add_argument("--fit", action="store_true",
                    help="駒種価値テーブルの順序ペア学習 (gp_fit)")
    ap.add_argument("--kif-dir", default=None)
    ap.add_argument("--bin", dest="fit_bin", default=None)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--wr-pairs", type=int, default=50000,
                    help="[WR-PAIR] eval由来の第2制約ペア数 (0=無効)")
    ap.add_argument("--calibrate", metavar="BIN",
                    help="ply 較正テーブルを再生成して表示 (gp_core.py へ手動反映)")
    ap.add_argument("--engine", default="Pyfamate.py")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    if a.sfen:
        print(f"GP = {gp_of_sfen(a.sfen, ply=a.ply):.6f}")
        return
    if a.fit:
        gp_fit(kif_dir=a.kif_dir, bin_path=a.fit_bin, iters=a.iters,
               wr_pairs=a.wr_pairs)
        return
    if a.verify_fast:
        import time as _t
        t0=_t.perf_counter(); n=0; mism=0
        for raw in iter_psv(a.verify_fast):
            data32 = struct.unpack(PSV_FMT, raw)[0]
            sb = SvBoard(); _fast_decode_psfen_into(data32, sb)
            if tuple(features_from_cells(sb)) != tuple(
                    _game_phase_state_counts_core(MiniBoard(_svboard_to_sfen(sb)))):
                mism += 1
            n += 1
        print(f"[verify-fast] n={n} mismatch={mism} ({_t.perf_counter()-t0:.1f}s)")
        sys.exit(0 if mism == 0 and n > 0 else 1)
    if a.calibrate:
        import statistics as _st
        bins = {}
        for raw in iter_psv(a.calibrate):
            data32, _s, _m, ply, _r, _p = struct.unpack(PSV_FMT, raw)
            sb = SvBoard(); _fast_decode_psfen_into(data32, sb)
            gp0 = gp_of_features(features_from_cells(sb), 0.0)
            bins.setdefault(int(gp0 / 0.05), []).append(ply)
        tbl = []; prev = 0
        for k in range(max(bins) + 1):
            v = bins.get(k)
            m = _st.median(v) if v else prev
            tbl.append(round(float(m), 1)); prev = m
        print("_PLY_CALIB =", tuple(tbl))
        return
    if a.verify:
        import importlib.util, os, time
        spec = importlib.util.spec_from_file_location("Pyfamate", os.path.abspath(a.engine))
        P = importlib.util.module_from_spec(spec)
        sys.modules["Pyfamate"] = P
        spec.loader.exec_module(P)
        cfg = P.EngineConfig.from_cfg(dict(P._CONFIG_DEFAULTS))
        total = os.path.getsize(a.verify) // PSV_RECORD_SIZE
        stride = max(1, total // a.n)
        diff_max = 0.0
        n = bad = 0
        t0 = time.perf_counter()
        with open(a.verify, "rb") as f:
            for k in range(min(a.n, total)):
                f.seek(k * stride * PSV_RECORD_SIZE)
                raw = f.read(PSV_RECORD_SIZE)
                if len(raw) < PSV_RECORD_SIZE:
                    break
                data32, score, _mv, gameply, _res, _pad = struct.unpack(PSV_FMT, raw)
                try:
                    g_std = gp_of_record(raw)[0]
                    sfen = P._fast_psfen_to_sfen(data32)
                    b = P.ShogiBoard(sfen="sfen " + sfen, track_history=False)
                    wh, pr, ca, ki, ex = P._game_phase_state_counts_core(b)
                    g_ref = _clamp(P._calc_state_phase(wh, pr, ca, ki, ex,
                                                       cfg=cfg, ply=gameply), 0.0, 1.5)
                except Exception:
                    bad += 1
                    continue
                diff_max = max(diff_max, abs(g_std - g_ref))
                n += 1
        print(f"[verify] n={n} bad={bad} max|Δgp|={diff_max:.9f} "
              f"({time.perf_counter()-t0:.1f}s)")
        sys.exit(0 if diff_max < 1e-9 and n > 0 else 1)
    ap.print_help()


def _cli_torch():
    import argparse
    ap = argparse.ArgumentParser(description="GP fit/grouping GPU レーン")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("torch-fit")
    f.add_argument("--kif-dir")
    f.add_argument("--bin")
    f.add_argument("--bin-sample", type=int, default=100000)
    f.add_argument("--iters", type=int, default=3000)
    f.add_argument("--lr", type=float, default=0.01)
    f.add_argument("--lam", type=float, default=0.005)
    f.add_argument("--pairs-per-game", type=int, default=400)
    f.add_argument("--bin-pairs", type=int, default=200000)
    f.add_argument("--min-dply-kif", type=int, default=8)
    f.add_argument("--min-dply-bin", type=int, default=40)
    f.add_argument("--wr-pairs", type=int, default=200000,
                   help="[WR-PAIR] eval 由来の第2制約ペア数 (0=無効)")
    f.add_argument("--wr-weight", type=float, default=0.25)
    f.add_argument("--wr-dply", type=int, default=10)
    f.add_argument("--wr-dcp", type=float, default=800.0)
    f.add_argument("--learn", default="pieces", choices=("pieces", "full"))
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--device", default="auto")
    f.add_argument("--out", default="gp_fit_result.json")
    g = sub.add_parser("torch-group")
    g.add_argument("bin")
    g.add_argument("--edges", default=_DEFAULT_GROUP_EDGES)
    g.add_argument("--ply-mode", default="state",
                   choices=("record", "zero", "state", "auto"))
    g.add_argument("--tables", default=None,
                   help="gp_fit_result.json を読み学習済みテーブルで棲み分け")
    g.add_argument("--report-only", action="store_true")
    g.add_argument("--require-done", action="store_true",
                   help="ディレクトリ指定時、.done マーカー付き bin のみ処理")
    g.add_argument("--split-bytes", type=int, default=_DEFAULT_SPLIT_BYTES,
                   help="バケット bin を N bytes 単位で分割 (0=無効, 既定≈19GB)")
    g.add_argument("--out-dir", default=None)
    g.add_argument("--device", default="auto")
    v = sub.add_parser("torch-verify")
    v.add_argument("bin")
    v.add_argument("--n", type=int, default=5000)
    v.add_argument("--device", default="auto")
    a = ap.parse_args()
    {"torch-fit": _torch_cmd_fit, "torch-group": _torch_cmd_group,
     "torch-verify": _torch_cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    _cli()
