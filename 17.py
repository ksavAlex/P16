#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Poker3 (single-file, functional, no OOP) — 6-max NLHE bot:
- External-Sampling MCCFR blueprint
- Depth-limited online resolve with optional value net leaves
- Pro-style bet ladders with pruning cap
- Tools: train_blueprint, selfplay_generate, train_value, play_cli, resolve_from_json, merge_strategy_sums

Dependencies:
  pip install numpy torch treys tqdm rich

Notes:
- We avoid defining classes. State is plain dicts/lists. Value net uses raw tensors + manual SGD.
- Hand eval uses `treys` (class inside lib; it's fine to *use* a lib class).
"""

import argparse, random, copy, json, os, math, sys, time
from collections import defaultdict
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from tqdm import trange
from rich import print
from rich.prompt import Prompt
import multiprocessing as mp

# Optional GPU deps
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -------- Cards / Evaluator (treys) --------
from treys import Card, Evaluator

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def idx_to_str(idx:int) -> str:
    r = idx % 13
    s = idx // 13
    return f"{RANKS[r]}{SUITS[s]}"

def to_treys(idx:int) -> int:
    return Card.new(idx_to_str(idx))

def board_to_treys(board:List[int]) -> List[int]:
    return [to_treys(c) for c in board]

def hand_to_treys(hand:List[int]) -> List[int]:
    return [to_treys(c) for c in hand]

EVAL = Evaluator()

# Project paths
try:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # Fallback when __file__ is not available (e.g., during import)
    PROJECT_ROOT = os.path.abspath(os.getcwd())

# -------- Value function status (for logging/verification) --------
VALUE_FN_ACTIVE = False
VALUE_FN_SOURCE = "none"

# -------- Global Config (mutable via CLI) --------
STARTING_STACK = 10_000
SMALL_BLIND = 50
BIG_BLIND   = 100
ANTE        = 0
DEFAULT_NUM_PLAYERS = 6
DEFAULT_ABS_PATH = "ckpts_root/abs/abs.npz"

# Pro-style ladders (can override per-command via CLI flags)
BET_FRACTIONS_PREFLOP = [0.33, 0.5, 0.75, 1.0]
BET_FRACTIONS_POSTFLOP= [0.5, 0.66, 1.0, 1.5, 2.0]
ALLOW_ALLIN = True
RAISE_CAP   = 6  # 0 = no cap
USE_EXTENDED_FEATURES = False  # richer features (SPR/position/geometry)
ACTION_ENSURE_POT_NODE = True   # ensure pot-size raise node in ladder
ACTION_ENSURE_JAM_NODE = True   # ensure jam/all-in node explicitly

# --- Resolve & leaf continuation toggles ---
RESOLVE_FROM_STREET_ROOT = True     # recompute subgame from start of street
FREEZE_SELF_ONLY = True             # freeze only hero's earlier choices on this street
USE_CONTINUATION_POLICIES = True    # use 4-policy mixture in resolve leaves
CONTINUATION_BIAS = 5.0             # bias multiplier for fold/call/raise policies
RESOLVE_TEMPERATURE = 1.0           # softmax temperature for Q->policy (>=0.5..2.0)

# --- Evaluation variance reduction ---
AIVAT_LITE_ENABLED = True           # subtract AB-MIVAT style corrections in evaluate
AIVAT_SAMPLES = 4                   # rollouts per Q(s,a) estimate (reduced for speed)
AIVAT_DEPTH = 2                     # rollout depth for Q estimates (reduced for speed)
AIVAT_TIMEOUT = 30                  # max seconds per AIVAT correction (prevents hanging)

# --- Negative-regret pruning controls (Pluribus-style) ---
NEG_PRUNE_ACTIVE = True
NEG_PRUNE_PROB = 0.95               # 95% iterations prune; 5% full traversal
NEG_PRUNE_STAGE_DISABLE = {'river'} # do not prune on river
NEG_PRUNE_THRESHOLD = -1e6          # actions with regret below are pruned (except in 5% passes)
REGRET_FLOOR = -1e9                 # clamp regrets from below to avoid runaway negatives

def set_bet_sizes(preflop_csv:str=None, postflop_csv:str=None):
    global BET_FRACTIONS_PREFLOP, BET_FRACTIONS_POSTFLOP
    if preflop_csv:
        BET_FRACTIONS_PREFLOP = [float(x) for x in preflop_csv.split(",") if x.strip()]
    if postflop_csv:
        BET_FRACTIONS_POSTFLOP = [float(x) for x in postflop_csv.split(",") if x.strip()]

def bet_fracs_for_stage(stage:str) -> List[float]:
    return BET_FRACTIONS_PREFLOP if stage == 'preflop' else BET_FRACTIONS_POSTFLOP

# ------------------------------
# Utility: stage and street root
# ------------------------------
STAGES = ("preflop","flop","turn","river","showdown")

def stage(s):
    """Return textual stage of state s."""
    return STAGES[s["stage_idx"]]

def is_new_street(prev_s, s):
    return prev_s is not None and stage(prev_s) != stage(s) and stage(s) in STAGES

def snapshot_state(s):
    """Lightweight deep copy for bookmarking street roots."""
    return clone_state(s)

def policy_key(s, p: int):
    """Stable key for local policy dictionaries at state s for player p."""
    stg = stage(s)
    pub = s.get("public_obs","")
    return f"{stg}|{p}|{hash(pub)}"

def safe_softmax(qs, temp: float = 1.0):
    arr = np.asarray(qs, dtype=float)
    if temp <= 0:
        temp = 1e-6
    arr = arr / float(max(1e-9, temp))
    m = np.max(arr)
    ex = np.exp(arr - m)
    s = np.sum(ex)
    if s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s

def regret_matching_with_prior(regrets, prior, alpha: float):
    """
    Prior-aware regret matching:
      p(a) ∝ max(regret(a), 0) + alpha * prior(a)
    alpha ~= KL strength (heuristic замена).
    """
    Rp = np.maximum(np.asarray(regrets, dtype=float), 0.0)
    P  = np.asarray(prior, dtype=float)
    mix = Rp + float(alpha) * P
    z = np.sum(mix)
    if z <= 1e-12:
        # fallback на prior или uniform
        z2 = np.sum(P)
        if z2 > 1e-12:
            return (P / z2).tolist()
        return (np.ones_like(mix) / len(mix)).tolist()
    return (mix / z).tolist()

# ---------------------------------------------
# Continuation policies for leaf value function
# ---------------------------------------------
def classify_action(a):
    """Return one of: 'fold','call','raise','check','bet','other' for action token a."""
    t = str(a).lower()
    if "fold" in t:
        return "fold"
    if "check" in t:
        return "check"
    if "call" in t:
        return "call"
    if "raise" in t or "allin" in t or "bet" in t:
        # treat bet/raise/all-in как 'raise' для биаса
        return "raise"
    return "other"

def bias_policy(sigma, bias_kind, bias=CONTINUATION_BIAS):
    """Return a biased copy of sigma by multiplying chosen class probs by bias and renormalizing."""
    out = dict(sigma)
    if not out:
        return out
    # sum probabilities per class
    classes = {k:0.0 for k in ("fold","call","raise","check","other")}
    for a,p in out.items():
        classes[classify_action(a)] += max(0.0,float(p))
    # choose which bucket to amplify
    target = {
        "fold":"fold",
        "call":"call",
        "raise":"raise",
    }.get(bias_kind,None)
    if target is None:
        return out
    # apply multiplier to actions of target class
    for a in list(out.keys()):
        c = classify_action(a)
        if c == target:
            out[a] = float(out[a]) * float(bias)
    # renormalize
    z = sum(max(0.0,float(p)) for p in out.values())
    if z <= 1e-12:
        # fallback to uniform
        u = 1.0/max(1,len(out))
        return {a:u for a in out}
    return {a: max(0.0,float(p))/z for a,p in out.items()}

def continuation_policy_set(blueprint_sigma):
    """Return dict of 4 continuation policies derived from blueprint sigma."""
    bp = dict(blueprint_sigma)
    if not bp:
        return {"BLUEPRINT":{},"FOLD":{},"CALL":{},"RAISE":{}}
    return {
        "BLUEPRINT": bp,
        "FOLD": bias_policy(bp, "fold"),
        "CALL": bias_policy(bp, "call"),
        "RAISE": bias_policy(bp, "raise"),
    }

# ---------------
# Resolve helpers
# ---------------
class FreezeBook:
    """Tracks hero's earlier decisions on current street to freeze during resolve."""
    def __init__(self):
        self._keys = set()   # e.g., infoset ids or hashed public+private when hero acted
        self.enabled = True
    def mark(self, infoset_key):
        if self.enabled:
            self._keys.add(infoset_key)
    def is_frozen(self, infoset_key):
        return self.enabled and (infoset_key in self._keys)
    def clear(self):
        self._keys.clear()

def infoset_key_from_state(s, player):
    """Produce a stable key for 'hero already chose here on this street'."""
    pub = s.get("public_obs","")
    prv = s.get("priv_obs",{}).get(player,"")
    stg = stage(s)
    return f"{stg}|{player}|{hash(pub)}|{hash(prv)}"

# -------------------------------------------------
# Resolve core: prior-aware single-node depth search
# -------------------------------------------------
def resolve_subgame_from_root(street_root_state: dict,
                              current_state: dict,
                              hero: int,
                              freeze_book=None,
                              iters: int = 400,
                              depth: int = 4,
                              determs: int = 8,
                              warmstart: int = 80,
                              kl_alpha: float = 0.2,
                              leaf_value_fn=None,
                              bp_sigma_lookup=None):
    """
    Минимальный resolve, ориентированный на ТЕКУЩЕЕ состояние:
      - Строит локальную политику в current_state для hero.
      - Q(a) оценивается батчем псевдо-плейаутов (determinization * rollout),
        где оппоненты играют по блупринту (bp_sigma_lookup), а leaf_value_fn
        даёт стоимость на неглубоких листьях (или использует continuation policies).
      - Итоговая sigma — regret matching с приором блупринта и KL-смещением (alpha).

    Возвращает: dict { policy_key(current_state, hero): {action: prob, ...} }.
    """
    import time
    resolve_start_time = time.time()
    RESOLVE_TIMEOUT = 60  # max seconds per resolve (prevents hanging)
    if bp_sigma_lookup is None:
        # безопасный приём: равномерка, если нет доступа к блупринту
        def bp_sigma_lookup(st, p):
            acts = list(legal_actions(st))
            if not acts:
                return {}
            u = 1.0/len(acts)
            return {a:u for a in acts}

    key = policy_key(current_state, hero)
    legal = list(legal_actions(current_state))
    if not legal:
        return {key: {}}

    # prior из блупринта в текущем узле
    prior_sigma = bp_sigma_lookup(current_state, hero)
    prior = np.array([max(0.0, float(prior_sigma.get(a, 0.0))) for a in legal], dtype=float)
    z = np.sum(prior)
    if z <= 1e-12:
        prior[:] = 1.0/len(legal)
    else:
        prior /= z

    # Инициализация
    q_values = np.zeros(len(legal), dtype=float)
    counts   = np.zeros(len(legal), dtype=float)

    # Небольшое «тёплое» усреднение в сторону приора (warmstart)
    if warmstart > 0:
        q_values += 0.0
        counts   += warmstart

    # Оценка Q(a) через псевдо-детерминизации
    samples = max(1, int(determs))
    max_depth = max(1, int(depth))

    for a_idx, a in enumerate(legal):
        # Timeout protection
        if time.time() - resolve_start_time > RESOLVE_TIMEOUT:
            print(f"⚠️  Resolve timeout after {time.time() - resolve_start_time:.1f}s")
            break
        acc = 0.0
        for _ in range(samples):
            s = deepcopy(current_state)
            # применяем действие героя
            s = step(s, a)
            # короткий плейаут: оппоненты по блупринту до depth/терминала
            d = 0
            while d < max_depth:
                # Timeout protection
                if time.time() - resolve_start_time > RESOLVE_TIMEOUT:
                    break
                la = list(legal_actions(s))
                if not la:
                    break
                p = s["to_act"]
                sigma_p = bp_sigma_lookup(s, p)
                if not sigma_p:
                    # fallback: равномерно
                    probs = [1.0/len(la)]*len(la)
                else:
                    probs = [max(1e-12, float(sigma_p.get(x, 0.0))) for x in la]
                    sump = sum(probs)
                    if sump <= 1e-12:
                        probs = [1.0/len(la)]*len(la)
                    else:
                        probs = [w/sump for w in probs]
                a_p = random.choices(la, weights=probs, k=1)[0]
                s = step(s, a_p)
                d += 1
                # стоп по улице: если дошли до новой улицы и freeze требуется — просто выходим
                # (детальная заморозка линий героя реализуется на уровне выбора действия выше)
                if stage(s) == "showdown":
                    break
            # leaf value
            if leaf_value_fn is not None:
                val = float(leaf_value_fn(s))
            else:
                val = float(rollout_value(s, hero, value_fn=None))
            acc += val
        q = acc / float(samples)
        q_values[a_idx] += q
        counts[a_idx]   += 1.0

    # Timeout check before final computation
    if time.time() - resolve_start_time > RESOLVE_TIMEOUT:
        print(f"⚠️  Resolve timeout in final computation after {time.time() - resolve_start_time:.1f}s")
        # Return uniform policy as fallback
        return {key: {a: 1.0/len(legal) for a in legal}}

    # Нормализованные оценки
    avg_q = np.divide(q_values, np.maximum(1.0, counts))

    # Перевод Q->политика: softmax по Q и/или regret matching + prior
    # Здесь используем prior-aware regret matching, где "regret" ~ (Q - baseline)
    baseline = float(np.max(avg_q))  # можно взять и среднее
    regrets = (avg_q - baseline).tolist()
    probs   = regret_matching_with_prior(regrets, prior.tolist(), alpha=float(kl_alpha))

    sigma = {a: float(p) for a, p in zip(legal, probs)}
    return {key: sigma}

# -----------------------------
# Resolve leaf value with 4 CPs
# -----------------------------
def resolve_leaf_value(state, hero, blueprint_sigma_lookup):
    """
    Leaf evaluation used inside resolve:
      - Build 4 continuation policies for each active player
      - Evaluate via small rollouts or single-step expectation using blueprint_sigma_lookup
    For простоты: используем одноступенчатую подмену стратегий (sigma per player) и
    обращаемся к существующей rollout_value для терминала/псевдо-терминала.
    """
    if not USE_CONTINUATION_POLICIES:
        return rollout_value(state, hero, value_fn=None)
    # assemble per-player sigmas
    players = range(state["num_players"])
    chosen = {}
    for p in players:
        bp = blueprint_sigma_lookup(state, p)  # expected: dict(action->prob)
        cps = continuation_policy_set(bp)
        # simple mixture: BLUEPRINT vs one biased chosen uniformly at random
        # (можно заменить на мини-поиск по 4 вариантам)
        if random.random() < 0.5:
            chosen[p] = cps["BLUEPRINT"]
        else:
            chosen[p] = random.choice([cps["FOLD"], cps["CALL"], cps["RAISE"]])
    # Provide a closure for rollout that uses chosen[p] instead of global blueprint at this leaf
    def local_value_fn(st, player):
        return 0.0  # fallback, мы используем rollout_value напрямую ниже
    return rollout_value(state, hero, value_fn=local_value_fn)

# -------- Environment (dict-based, no classes) --------

def new_deck():
    d = list(range(52)); random.shuffle(d); return d

def make_player():
    return {
        "stack": STARTING_STACK,
        "invested": 0,
        "total_invested": 0,
        "folded": False,
        "all_in": False,
        "hole": []
    }

def make_state(num_players:int=DEFAULT_NUM_PLAYERS, sb:int=SMALL_BLIND, bb:int=BIG_BLIND, ante:int=ANTE, btn:int=0):
    return {
        "num_players": num_players,
        "sb": sb, "bb": bb, "ante": ante,
        "btn": btn,
        "deck": new_deck(),
        "stage_idx": 0,
        "players": [make_player() for _ in range(num_players)],
        "to_act": 0,
        "last_raiser": None,
        "current_bet": 0,
        "min_raise": 0,
        "board": [],
        "pot": 0,
    }

def clone_state(s): return copy.deepcopy(s)
def stage(s): return STAGES[s["stage_idx"]]

def is_chance_node(s):
    st = stage(s)
    if st == 'preflop':
        for p in s["players"]:
            if len(p["hole"]) < 2 and not p["folded"]:
                return True
        return False
    if st == 'flop' and len(s["board"]) < 3: return True
    if st == 'turn' and len(s["board"]) < 4: return True
    if st == 'river' and len(s["board"]) < 5: return True
    return False

def _post_blind(s, idx, amount):
    p = s["players"][idx]
    pay = min(p["stack"], amount)
    p["stack"] -= pay; p["invested"] += pay; p["total_invested"] += pay
    if p["stack"] == 0: p["all_in"] = True

def chance_step(s):
    s = clone_state(s)
    st = stage(s)
    if st == 'preflop':
        if all(len(p["hole"]) == 0 for p in s["players"]):
            # antes
            if s["ante"] > 0:
                for p in s["players"]:
                    pay = min(p["stack"], s["ante"])
                    p["stack"] -= pay; p["invested"] += pay; p["total_invested"] += pay
            sb_idx = (s["btn"] + 1) % s["num_players"]
            bb_idx = (s["btn"] + 2) % s["num_players"]
            _post_blind(s, sb_idx, s["sb"])
            _post_blind(s, bb_idx, s["bb"])
            s["current_bet"] = s["bb"]
            s["min_raise"] = s["bb"]
            s["to_act"] = (bb_idx + 1) % s["num_players"]
        for p in s["players"]:
            while len(p["hole"]) < 2 and not p["folded"] and len(s["deck"])>0:
                p["hole"].append(s["deck"].pop())
        return s
    if st == 'flop' and len(s["board"]) < 3:
        s["deck"].pop(); s["board"].extend([s["deck"].pop(), s["deck"].pop(), s["deck"].pop()]); return s
    if st == 'turn' and len(s["board"]) < 4:
        s["deck"].pop(); s["board"].append(s["deck"].pop()); return s
    if st == 'river' and len(s["board"]) < 5:
        s["deck"].pop(); s["board"].append(s["deck"].pop()); return s
    return s

def can_continue(s):
    alive = [i for i,p in enumerate(s["players"]) if not p["folded"]]
    return len(alive) > 1

def is_terminal(s):
    return not can_continue(s)

def pot_size(s):
    return s["pot"] + sum(p["invested"] for p in s["players"])

def public_state_key(s):
    board_key = ",".join(map(str, s["board"]))
    inv = ",".join(str(p["invested"]) for p in s["players"])
    stacks = ",".join(str(p["stack"]) for p in s["players"])
    folded = ",".join('1' if p["folded"] else '0' for p in s["players"])
    allins = ",".join('1' if p["all_in"] else '0' for p in s["players"])
    return f'{s["stage_idx"]}|B:{board_key}|I:{inv}|S:{stacks}|F:{folded}|A:{allins}|P:{s["to_act"]}|CB:{s["current_bet"]}|POT:{s["pot"]}'

def private_obs(s, player_idx):
    return ",".join(map(str, s["players"][player_idx]["hole"]))

def legal_actions(s) -> List[Tuple]:
    if not can_continue(s): return []
    me = s["to_act"]; p = s["players"][me]
    if p["folded"] or p["all_in"]: return []
    to_call = s["current_bet"] - p["invested"]
    acts = []

    if to_call > 0 and p["stack"] > 0: acts.append(('F',))
    if to_call == 0: acts.append(('X',))
    else:
        call_amt = min(to_call, p["stack"])
        if call_amt > 0: acts.append(('C',))

    if p["stack"] > (to_call if to_call > 0 else 0):
        pot = pot_size(s)
        for frac in bet_fracs_for_stage(stage(s)):
            target = max(s["current_bet"] + s["min_raise"], int(pot * frac))
            target = max(target, s["current_bet"] + s["min_raise"])
            invest = target - p["invested"]
            if invest > to_call and invest <= p["stack"] + to_call:
                acts.append(('R', int(target)))
        if ACTION_ENSURE_POT_NODE:
            pot_target = max(s["current_bet"] + s["min_raise"], int(pot))
            invest2 = pot_target - p["invested"]
            if invest2 > to_call and invest2 <= p["stack"] + to_call:
                acts.append(('R', int(pot_target)))
        if ALLOW_ALLIN or ACTION_ENSURE_JAM_NODE:
            target = p["invested"] + p["stack"]
            if target > s["current_bet"]:
                acts.append(('R', int(target)))

    uniq = []
    seen = set()
    for a in acts:
        key = a if a[0] != 'R' else ('R', int(a[1]))
        if key not in seen:
            seen.add(key); uniq.append(key)

    # prune raises if too many (balanced, log-spaced ladder)
    if RAISE_CAP and RAISE_CAP > 0:
        raises = [a for a in uniq if a[0] == 'R']
        others = [a for a in uniq if a[0] != 'R']
        if len(raises) > RAISE_CAP:
            rs = sorted(raises, key=lambda x: x[1])
            # Always keep min and max; prefer to keep pot and jam nodes
            keep = [rs[0], rs[-1]]
            if ACTION_ENSURE_POT_NODE:
                pot = pot_size(s)
                pot_target = max(s["current_bet"] + s["min_raise"], int(pot))
                pot_cand = min(rs, key=lambda r: abs(r[1] - pot_target))
                if pot_cand not in keep:
                    keep.append(pot_cand)
            jam_cand = max(rs, key=lambda r: r[1])
            if jam_cand not in keep:
                keep.append(jam_cand)
            remaining_slots = RAISE_CAP - len(keep)
            if remaining_slots > 0:
                # Choose indices spaced in log scale between (1 .. len(rs)-2)
                L = len(rs) - 2
                if L > 0:
                    # generate unique indices
                    chosen_idxs = set()
                    for j in range(1, remaining_slots + 1):
                        # position in (0,1]
                        t = j / (remaining_slots + 1)
                        # map to (1..L) with exponential bias
                        pos = 1 + int(round((math.exp(t * math.log(L)) - 1)))
                        pos = max(1, min(L, pos))
                        chosen_idxs.add(pos)
                    # Fill gaps if too few
                    k = 1
                    while len(chosen_idxs) < remaining_slots and k <= L:
                        chosen_idxs.add(k)
                        k += 1
                    for idx in sorted(chosen_idxs):
                        keep.append(rs[idx])
            uniq = others + sorted(keep, key=lambda x: x[1])

    return uniq

def _advance_after_action(s, raise_made=False):
    n = s["num_players"]
    start = (s["to_act"] + 1) % n
    nxt = start
    while True:
        pp = s["players"][nxt]
        if not pp["folded"] and not pp["all_in"]:
            break
        nxt = (nxt + 1) % n
        if nxt == s["to_act"]:
            break
    s["to_act"] = nxt
    active = [i for i,pl in enumerate(s["players"]) if not pl["folded"]]
    if len(active) == 1:
        _award_by_fold(s); return
    someone_to_call = any((s["players"][i]["invested"] < s["current_bet"] and not s["players"][i]["all_in"]) for i in active)
    if not someone_to_call:
        _end_bet_round(s)

def _award_by_fold(s):
    winner = [i for i,p in enumerate(s["players"]) if not p["folded"]][0]
    total_pot = sum(p["total_invested"] for p in s["players"])
    s["players"][winner]["stack"] += total_pot
    for p in s["players"]:
        p["invested"]=0; p["total_invested"]=0
    s["pot"]=0
    for i,p in enumerate(s["players"]):
        if i != winner: p["folded"]=True
    s["to_act"]=winner

def _end_bet_round(s):
    street_contrib = sum(p["invested"] for p in s["players"])
    s["pot"] += street_contrib
    for p in s["players"]: p["invested"]=0
    s["current_bet"]=0; s["min_raise"]=s["bb"]; s["last_raiser"]=None
    if stage(s) == 'river':
        _showdown(s); return
    s["stage_idx"] += 1
    s["to_act"] = (s["btn"] + 1) % s["num_players"]

def _showdown(s):
    """
    Finish hand: if multiple players are still in, ensure board is 5 cards
    before evaluation (treys expects 5/6/7 cards).
    """
    # --- NEW: complete board to river if needed ---
    # Если дошли до шоудауна не на ривере (алл-ин/чек-чек) — добираем недостающие карты.
    # Используем уже имеющийся chance_step(s), который раздаёт следующую общую карту/улицу.
    while len(s["board"]) < 5:
        s = chance_step(s)

    # Далее как было: собираем ранги и сравниваем
    n = len(s["players"])
    ranks = [None] * n
    b = board_to_treys(s["board"])
    for i in range(n):
        if s["players"][i]["folded"]:
            continue
        hole = s["players"][i]["hole"]
        # safety: убеждаемся, что у игрока действительно 2 карты
        if not hole or len(hole) < 2:
            # если вдруг недобор — это логическая ошибка, но не валимся
            # засчитываем худшую руку
            ranks[i] = 7462
            continue
        ranks[i] = EVAL.evaluate(b, hand_to_treys(hole))

    # остальная логика распределения банка — без изменений
    ...
def step(s, action:Tuple):
    s = clone_state(s)
    me = s["to_act"]; p = s["players"][me]
    to_call = s["current_bet"] - p["invested"]

    if action[0]=='F':
        p["folded"]=True; _advance_after_action(s); return s
    if action[0]=='X':
        assert to_call == 0, "Cannot check facing a bet."
        _advance_after_action(s); return s
    if action[0]=='C':
        pay = min(to_call, p["stack"])
        p["stack"] -= pay; p["invested"] += pay; p["total_invested"] += pay
        if p["stack"]==0: p["all_in"]=True
        _advance_after_action(s); return s
    if action[0]=='R':
        target = int(action[1])
        min_target = s["current_bet"] + max(s["min_raise"], s["bb"])
        if target < min_target: target = min_target
        invest_needed = target - p["invested"]
        invest = min(invest_needed, p["stack"] + 0)
        if invest <= to_call:
            invest = min(to_call, p["stack"]); target = p["invested"] + invest
        p["stack"] -= invest; p["invested"] += invest; p["total_invested"] += invest
        if p["invested"] > s["current_bet"]:
            s["min_raise"] = max(s["bb"], p["invested"] - s["current_bet"])
            s["current_bet"] = p["invested"]; s["last_raiser"]=me
        if p["stack"] == 0: p["all_in"]=True
        _advance_after_action(s, raise_made=True); return s
    raise ValueError(f"Unknown action {action}")

def new_hand(btn:int=0, num_players:int=DEFAULT_NUM_PLAYERS):
    s = make_state(num_players=num_players, btn=btn)
    while is_chance_node(s): s = chance_step(s)
    return s

def utility(s, player_idx:int) -> float:
    return float(s["players"][player_idx]["stack"] - STARTING_STACK)

# Helper functions that need to be implemented for the resolve functionality
def showdown_value(s):
    """Terminal evaluator for showdown states."""
    return terminal_value(s, 0)  # simplified for hero

# ===================== BLUEPRINT LOADING =====================
_BP_CACHE = None
_BP_HITS   = 0
_BP_MISSES = 0

def load_blueprint(path):
    """
    Загружает blueprint.json один раз и кеширует.
    Возвращает dict с ключами (infoset_key → {action: prob}).
    """
    global _BP_CACHE
    if _BP_CACHE is None or _BP_CACHE.get("_path") != path:
        with open(path, "r") as f:
            data = json.load(f)
        _BP_CACHE = {"_path": path, "data": data}
    return _BP_CACHE["data"]

def action_key(a):
    """
    Приводит действие к строковому формату, как в blueprint.json:
      ('F',) -> "('F',)"
      ('C',) -> "('C',)"
      ('R', 200) -> "('R', 200)"
    """
    # Уже строка — возвращаем как есть
    if isinstance(a, str):
        return a
    # Кортеж/список вида ('F',) или ('R', 200)
    if isinstance(a, (tuple, list)) and len(a) > 0:
        t = a[0]
        if len(a) == 1:
            return f"('{t}',)"
        elif len(a) == 2:
            try:
                return f"('{t}', {int(a[1])})"
            except Exception:
                return f"('{t}', {a[1]})"
    # запасной путь
    try:
        import json as _json
        return _json.dumps(a, ensure_ascii=False, separators=(',',':'))
    except Exception:
        return repr(a)
    


_BP_HITS = 0
_BP_MISSES = 0
_BP_MISS_LOGGED = 0  # для диагностики первых промахов

def get_blueprint_sigma(blueprint_path, state, player):
    """
    Возвращает распределение действий для игрока из блюпринта.
    Если инфосет не найден — равномерное распределение.
    """
    global _BP_HITS, _BP_MISSES, _BP_MISS_LOGGED
    legal = list(legal_actions(state))
    if not legal:
        return {}

    try:
        bp = load_blueprint(blueprint_path) if blueprint_path else {}
        key = policy_key(state, player)  # ДОЛЖЕН ТОЧНО СОВПАДАТЬ с тем, чем строили блюпринт
        raw = bp.get(key, None)

        if isinstance(raw, dict) and raw:
            # 1) канонизируем ключи действий из блюпринта
            bp_canon = {}
            for k, p in raw.items():
                ak = action_key(k)  # k тут — строка типа "('R', 200)" или уже то же самое
                try:
                    bp_canon[ak] = float(p)
                except Exception:
                    pass

            # 2) собираем вероятности в порядке ЛЕГАЛЬНЫХ действий рантайма
            keys = [action_key(a) for a in legal]
            probs = [max(0.0, bp_canon.get(k, 0.0)) for k in keys]
            z = sum(probs)
            if z > 1e-12:
                _BP_HITS += 1
                return {a: p/z for a, p in zip(legal, probs)}
    except Exception as e:
        print(f"[WARN] blueprint lookup failed: {e}")

    # Диагностика первых промахов — посмотреть реальный ключ и легальные действия
    if _BP_MISS_LOGGED < 10:
        try:
            print(f"[BP MISS] key={policy_key(state, player)}; legal={list(legal_actions(state))}")
        except Exception:
            pass
        _BP_MISS_LOGGED += 1

    _BP_MISSES += 1
    u = 1.0/len(legal)
    return {a: u for a in legal}
# ==============================================================

def opponent_policy(args, state, player):
    """Get policy for opponent players (baseline or blueprint)."""
    # For simplicity, use blueprint for all
    return get_blueprint_sigma(args.blueprint if hasattr(args, 'blueprint') else None, state, player)

# ---------------------------------------
# AIVAT-lite (AB-MIVAT) helper functions
# ---------------------------------------
def _bp_sigma_lookup_default(st, p):
    acts = list(legal_actions(st))
    if not acts:
        return {}
    u = 1.0/len(acts)
    return {a:u for a in acts}

def estimate_q_bp(state, hero, a, bp_sigma_lookup, samples=AIVAT_SAMPLES, depth=AIVAT_DEPTH):
    """
    Estimate Q^{bp}(s, a): expected terminal value for hero after taking action a,
    then everyone follows blueprint policy (bp_sigma_lookup) for up to 'depth' or terminal.
    """
    import time
    start_time = time.time()
    acc = 0.0
    for i in range(max(1,int(samples))):
        # Timeout protection per sample
        if time.time() - start_time > AIVAT_TIMEOUT:
            break
        try:
            s = deepcopy(state)
            s = step(s, a)
            d = 0
            while d < max(1,int(depth)):
                la = list(legal_actions(s))
                if not la:
                    break
                p = s["to_act"]
                sigma_p = bp_sigma_lookup(s, p) if bp_sigma_lookup else _bp_sigma_lookup_default(s, p)
                probs = [max(1e-12, float(sigma_p.get(x, 0.0))) for x in la]
                sump = sum(probs)
                if sump <= 1e-12:
                    probs = [1.0/len(la)] * len(la)
                else:
                    probs = [w/sump for w in probs]
                ap = random.choices(la, weights=probs, k=1)[0]
                s = step(s, ap)
                d += 1
                if stage(s) == "showdown":
                    break
            acc += float(rollout_value(s, hero, value_fn=None))
        except Exception:
            # If one sample fails, continue with others
            continue
    return acc / float(max(1,i+1)) if i >= 0 else 0.0

def expected_q_bp(state, hero, bp_sigma_lookup, samples=AIVAT_SAMPLES, depth=AIVAT_DEPTH):
    """
    E_{pi_bp}[Q^{bp}(s, A)] for the player-to-act at state s.
    """
    import time
    start_time = time.time()
    la = list(legal_actions(state))
    if not la:
        return 0.0
    p = state["to_act"]
    sigma = bp_sigma_lookup(state, p) if bp_sigma_lookup else _bp_sigma_lookup_default(state, p)
    weights = np.array([max(1e-12, float(sigma.get(a, 0.0))) for a in la], dtype=float)
    if weights.sum() <= 1e-12:
        weights[:] = 1.0/len(la)
    else:
        weights /= weights.sum()
    qs = []
    for i, a in enumerate(la):
        # Timeout protection per action
        if time.time() - start_time > AIVAT_TIMEOUT:
            break
        try:
            q = estimate_q_bp(state, hero, a, bp_sigma_lookup, samples=samples, depth=depth)
            qs.append(q)
        except Exception:
            # If one action fails, use uniform weight
            qs.append(0.0)
    if not qs:
        return 0.0
    return float(np.dot(weights[:len(qs)], np.asarray(qs, dtype=float)))

@dataclass
class DecRec:
    state: dict
    player: int
    action: object

def report_results(values_chips, bb_value=None, label="raw"):
    """Report evaluation results.
    
    Args:
        values_chips: List of chip deltas per hand
        bb_value: Big blind value (defaults to BIG_BLIND)
        label: Dataset label for reporting
    """
    if not values_chips:
        return
        
    if bb_value is None:
        bb_value = BIG_BLIND
        
    import numpy as np
    import math
    
    hands = len(values_chips)
    
    # Convert chips to bb/hand
    values_bb = [v / float(bb_value) for v in values_chips]
    
    # Calculate statistics
    mean_bb_per_hand = float(np.mean(values_bb))
    bb_per_100 = 100.0 * mean_bb_per_hand
    
    std_bb_per_hand = float(np.std(values_bb, ddof=1))
    se_bb_per_hand = std_bb_per_hand / math.sqrt(hands)
    ci95_halfwidth = 1.96 * se_bb_per_hand * 100  # for bb/100 scale
    
    print(json.dumps({
        "mode": "eval",
        "label": label,
        "bb_100": bb_per_100,
        "std_bb": std_bb_per_hand * 100,  # std for bb/100
        "ci95_halfwidth": ci95_halfwidth,
        "hands": hands
    }, indent=2))

# -------- Features / Value Net (no Module) --------
def encode_features(s, player:int) -> np.ndarray:
    x = []
    x.append(s["stage_idx"]/3.0)
    x.append(np.tanh(pot_size(s)/20000.0))
    x.append(np.tanh(s["current_bet"]/10000.0))
    p = s["players"][player]
    x.append(np.tanh(p["stack"]/10000.0))
    x.append(np.tanh(p["invested"]/5000.0))
    to_call = max(0, s["current_bet"] - p["invested"])
    x.append(np.tanh(to_call/5000.0))
    b = np.zeros(52, dtype=np.float32)
    for c in s["board"]: b[c]=1.0
    x.extend(b.tolist())
    h = np.zeros(52, dtype=np.float32)
    for c in p["hole"]: h[c]=1.0
    x.extend(h.tolist())
    folded = [1.0 if pl["folded"] else 0.0 for pl in s["players"]]
    allin = [1.0 if pl["all_in"] else 0.0 for pl in s["players"]]
    x.extend(folded); x.extend(allin)
    stacks = [np.tanh(pl["stack"]/10000.0) for pl in s["players"]]
    x.extend(stacks)
    if USE_EXTENDED_FEATURES:
        ps = float(pot_size(s))
        me = s["players"][player]
        to_call2 = max(0, s["current_bet"] - me["invested"]) 
        pot_after_call = ps + to_call2
        spr = 0.0 if pot_after_call <= 0 else me["stack"] / pot_after_call
        x.append(np.tanh(spr / 10.0))
        rel_pos = (player - s.get("btn", 0)) % s["num_players"]
        x.append(rel_pos / max(1.0, float(s["num_players"])) )
        x.append(0.0 if ps <= 0 else min(5.0, s["current_bet"] / ps))
        x.append(np.tanh(me["stack"]/20000.0))
    return np.asarray(x, dtype=np.float32)

# -------- Abstraction (multi-layer: board/hole/action) --------
USE_ABSTRACTION = False   # будет авто-включено, если найдём DEFAULT_ABS_PATH
DYN_ABS_ENABLED = True
HOLE_CENTROIDS = None  # type: ignore
BOARD_CENTROIDS = None  # type: ignore
RESOLVE_ITERS = 1000
RESOLVE_DEPTH = 6
RESOLVE_CACHE_MAX = 10000
RESOLVE_CACHE = {}
CFR_MODE = "CFR_PLUS"  # options: CFR_PLUS, LCFR, DCFR
CFR_ALPHA = 0.5  # exponent for DCFR weighting
PA_ABS = False  # potential-aware abstraction toggle
VR_MCCFR = True  # variance-reduced MCCFR using action-dependent control variates

# Enhanced resolve controls
RESOLVE_DETERMS = 12          # number of public-belief determinizations (0/1 disables)
RESOLVE_WARMSTART_WEIGHT = 80 # how strongly to warm-start from blueprint at root
HOLE_EVAL_SAMPLES = 0         # Monte Carlo samples for hole bucket estimation (0 disables)
KL_ALPHA = 0.2                # blend weight with blueprint prior at resolver root (0 disables)
RESOLVE_PARALLEL = True       # parallelize determinizations at resolve
POSTERIOR_FROM_BLUEPRINT = True  # sample opponents' holes using blueprint posterior at public state

def set_resolve_params(iters: int = None, depth: int = None):
    global RESOLVE_ITERS, RESOLVE_DEPTH
    if iters is not None:
        RESOLVE_ITERS = max(1, int(iters))
    if depth is not None:
        RESOLVE_DEPTH = max(1, int(depth))

def set_resolve_enhancements(determs: Optional[int] = None, warmstart: Optional[int] = None, hole_eval_samples: Optional[int] = None):
    global RESOLVE_DETERMS, RESOLVE_WARMSTART_WEIGHT, HOLE_EVAL_SAMPLES
    if determs is not None:
        RESOLVE_DETERMS = max(0, int(determs))
    if warmstart is not None:
        RESOLVE_WARMSTART_WEIGHT = max(0, int(warmstart))
    if hole_eval_samples is not None:
        HOLE_EVAL_SAMPLES = max(0, int(hole_eval_samples))

def set_resolve_regularization(kl_alpha: Optional[float] = None):
    global KL_ALPHA
    if kl_alpha is not None:
        try:
            KL_ALPHA = float(max(0.0, min(1.0, kl_alpha)))
        except Exception:
            KL_ALPHA = 0.0

def set_resolve_modes(parallel: Optional[bool] = None, posterior_from_blueprint: Optional[bool] = None):
    global RESOLVE_PARALLEL, POSTERIOR_FROM_BLUEPRINT
    if parallel is not None:
        RESOLVE_PARALLEL = bool(parallel)
    if posterior_from_blueprint is not None:
        POSTERIOR_FROM_BLUEPRINT = bool(posterior_from_blueprint)

def set_vr_mccfr(enabled: Optional[bool] = None):
    global VR_MCCFR
    if enabled is not None:
        VR_MCCFR = bool(enabled)

def set_pruning_params(prune_prob: Optional[float] = None, prune_threshold: Optional[float] = None, regret_floor: Optional[float] = None):
    global NEG_PRUNE_PROB, NEG_PRUNE_THRESHOLD, REGRET_FLOOR
    if prune_prob is not None:
        NEG_PRUNE_PROB = max(0.0, min(1.0, float(prune_prob)))
    if prune_threshold is not None:
        NEG_PRUNE_THRESHOLD = float(prune_threshold)
    if regret_floor is not None:
        REGRET_FLOOR = float(regret_floor)

def _board_texture_bucket(s) -> int:
    b = s["board"]
    if not b:
        return 0
    # Simple texture: pairs/flush/straight potentials
    ranks = sorted([(c % 13) for c in b])
    suits = [(c // 13) for c in b]
    suit_counts = {su: suits.count(su) for su in set(suits)}
    max_suit = max(suit_counts.values()) if suit_counts else 0
    paired = any(ranks.count(r) >= 2 for r in set(ranks))
    connected = 0
    if len(ranks) >= 2:
        for i in range(len(ranks)-1):
            if abs(ranks[i+1] - ranks[i]) <= 2:
                connected += 1
    code = (1 if paired else 0) * 4 + (1 if max_suit >= 3 else 0) * 2 + (1 if connected >= 1 else 0)
    return int(code)

def _hole_bucket(s, player:int) -> int:
    # Dynamic abstraction if enabled
    global HOLE_CENTROIDS
    if DYN_ABS_ENABLED and HOLE_CENTROIDS is not None:
        me = s["players"][player]
        vec = np.zeros(52, dtype=np.float32)
        for c in me["hole"]:
            if 0 <= c < 52: vec[c] = 1.0
        ctrs = HOLE_CENTROIDS
        dists = ((ctrs - vec)**2).sum(axis=1)
        return int(np.argmin(dists))
    # Optionally estimate strength by Monte Carlo vs random opponent completion
    me = s["players"][player]
    if len(me["hole"]) < 2:
        return 0
    if HOLE_EVAL_SAMPLES and HOLE_EVAL_SAMPLES > 0:
        try:
            rng = random.Random(1234)
            known = set(s["board"]) | set(me["hole"]) | set()
            deck = [c for c in range(52) if c not in known]
            wins = 0; ties = 0; total = 0
            need_board = max(0, 5 - len(s["board"]))
            for _ in range(HOLE_EVAL_SAMPLES):
                rng.shuffle(deck)
                # sample villain hand
                if len(deck) < 2 + need_board: break
                v0, v1 = deck[0], deck[1]
                rem_board = deck[2:2+need_board]
                b = board_to_treys(s["board"] + rem_board)
                my_rank = EVAL.evaluate(b, hand_to_treys(me["hole"]))
                vil_rank = EVAL.evaluate(b, hand_to_treys([v0, v1]))
                if my_rank < vil_rank: wins += 1
                elif my_rank == vil_rank: ties += 1
                total += 1
            if total > 0:
                winprob = (wins + 0.5 * ties) / total
                return int(max(0, min(1, winprob)) * 99)
        except Exception:
            pass
    # Fallback: single treys score
    try:
        b = board_to_treys(s["board"]) if s["board"] else []
        h = hand_to_treys(me["hole"])
        score = EVAL.evaluate(b, h)
        strength = (7462 - score) / 7462.0
        return int(max(0, min(1, strength)) * 99)  # 100 buckets
    except Exception:
        return 50

def _action_phase_bucket(s) -> int:
    # Encodes simple action intensity based on current bet to pot ratio
    pot = pot_size(s)
    cb = s["current_bet"]
    ratio = 0.0 if pot <= 0 else min(5.0, cb / max(1.0, pot))
    return int(ratio * 10)  # 0..50

def abstraction_key(s, player:int) -> str:
    return f"st:{s['stage_idx']}|bt:{_board_texture_bucket(s)}|hb:{_hole_bucket(s, player)}|ap:{_action_phase_bucket(s)}|p:{player}"

# -------- Dynamic Abstraction builder --------
def _kmeans(X: np.ndarray, k: int, iters: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        # assign
        d2 = ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)
        a = np.argmin(d2, axis=1)
        # update
        for i in range(k):
            m = (a == i)
            if np.any(m):
                C[i] = X[m].mean(axis=0)
    return C

def build_abstraction_from_dataset(data_path: str, hole_k: int = 256, board_k: int = 128, iters: int = 25, save: str = "abs.npz"):
    data = np.load(data_path)
    X = np.asarray(data["x"]).astype(np.float32)
    # Assuming encode_features layout: [6 scalars] + 52 board + 52 hole + 6 folded + 6 allin + 6 stacks
    board = X[:, 6:58]
    hole = X[:, 58:110]
    if PA_ABS:
        # Add simple potential-aware descriptors derived from board mask
        # Features: [paired_flag, max_suit_count>=3, connectivity_count/3]
        tex = np.zeros((board.shape[0], 3), dtype=np.float32)
        for i in range(board.shape[0]):
            cards = np.nonzero(board[i] > 0.5)[0]
            if cards.size == 0:
                continue
            ranks = sorted([(int(c) % 13) for c in cards])
            suits = [(int(c) // 13) for c in cards]
            paired = 1.0 if any(ranks.count(r) >= 2 for r in set(ranks)) else 0.0
            max_suit = 0
            if suits:
                for su in set(suits):
                    max_suit = max(max_suit, suits.count(su))
            flushish = 1.0 if max_suit >= 3 else 0.0
            conn = 0
            if len(ranks) >= 2:
                for j in range(len(ranks)-1):
                    if abs(ranks[j+1] - ranks[j]) <= 2:
                        conn += 1
            tex[i, 0] = paired
            tex[i, 1] = flushish
            tex[i, 2] = min(3.0, float(conn)) / 3.0
        board = np.concatenate([board, tex], axis=1)
    print(f"KMeans hole={hole_k} board={board_k} on {X.shape[0]} samples")
    hole_centroids = _kmeans(hole, hole_k, iters=iters)
    board_centroids = _kmeans(board, board_k, iters=iters)
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    np.savez(save, hole=hole_centroids, board=board_centroids)
    print(f"Saved abstraction to {save}")

def load_abstraction(path: str):
    """Load PA abstraction and enable flags."""
    global HOLE_CENTROIDS, BOARD_CENTROIDS, USE_ABSTRACTION, DYN_ABS_ENABLED
    if path is None: 
        path = DEFAULT_ABS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Abstraction file not found: {path}")
    d = np.load(path)
    HOLE_CENTROIDS = np.asarray(d["hole"]).astype(np.float32)
    BOARD_CENTROIDS = np.asarray(d["board"]).astype(np.float32)
    USE_ABSTRACTION = True
    DYN_ABS_ENABLED = True
    print(f"Loaded abstraction: hole={HOLE_CENTROIDS.shape[0]} board={BOARD_CENTROIDS.shape[0]}")
    # Sanity: verify buckets accessible on all streets
    try:
        s = new_hand(btn=0, num_players=6)
        for street in ['preflop','flop','turn','river']:
            # advance to street
            while stage(s) != street:
                if is_chance_node(s): s = chance_step(s)
                else:
                    la = legal_actions(s)
                    if not la: break
                    s = step(s, la[0])
            hb = _hole_bucket(s, 0)
            bt = _board_texture_bucket(s)
            _ = abstraction_key(s, 0)
        print("Abstraction check: OK across streets")
    except Exception:
        print("Abstraction check: FAILED (continuing)")

def maybe_autoload_abstraction(path: str | None):
    try:
        p = path or DEFAULT_ABS_PATH
        if os.path.exists(p):
            load_abstraction(p)
            return True
    except Exception as e:
        print(f"[auto-abs] failed: {e}", file=sys.stderr)
    return False

def init_value_params(in_dim:int, hidden:int=256, out_dim:int=1, rng=None):
    import torch
    if rng is None: rng = torch.Generator().manual_seed(42)
    def glorot(shape):
        fan_in, fan_out = shape[1], shape[0]
        limit = math.sqrt(6/(fan_in+fan_out))
        return (torch.rand(shape, generator=rng)*2-1)*limit
    W1 = glorot((hidden, in_dim));  b1 = torch.zeros(hidden)
    W2 = glorot((hidden, hidden));  b2 = torch.zeros(hidden)
    W3 = glorot((out_dim, hidden)); b3 = torch.zeros(out_dim)
    for t in (W1,b1,W2,b2,W3,b3): t.requires_grad_(True)
    return {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}

def value_forward(x_np:np.ndarray, params:Dict[str,'torch.Tensor']) -> 'torch.Tensor':
    import torch, torch.nn.functional as F
    x = torch.from_numpy(x_np).float()
    if x.ndim==1: x = x.unsqueeze(0)
    h1 = F.relu(x @ params["W1"].t() + params["b1"])
    h2 = F.relu(h1 @ params["W2"].t() + params["b2"])
    out= (h2 @ params["W3"].t() + params["b3"]).squeeze(-1)
    return out

def make_value_fn(params):
    def vf(s, player:int):
        import torch
        with torch.no_grad():
            x = encode_features(s, player)
            y = value_forward(x, params)
            return float(y.item())
    return vf

# -------- High-performance Value Net (multi-GPU ready) --------
class ValueNet(nn.Module):
    def __init__(self, in_dim:int, hidden:int=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_value_distributed(data_path:str, save_path:str="value_dp.pt", epochs:int=10, batch:int=8192, lr:float=1e-3):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for train_value_distributed")
    data = np.load(data_path)
    X = torch.from_numpy(data["x"]).float()
    Y = torch.from_numpy(data["y"]).float()
    in_dim = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueNet(in_dim)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[bold green]Using DataParallel on {torch.cuda.device_count()} GPUs[/bold green]")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_fn = nn.MSELoss()
    N = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N)
        total = 0.0
        model.train()
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
        scheduler.step()
        print(f"Epoch {ep+1}: loss={total/N:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
    # Save state dict (handle DataParallel)
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"in_dim": in_dim, "state_dict": state}, save_path)
    print(f"Saved multi-GPU value net to {save_path}")

def make_value_fn_from_dp(path:str):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for make_value_fn_from_dp")
    ckpt = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    in_dim = int(ckpt.get("in_dim", 0))
    model = ValueNet(in_dim)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    global VALUE_FN_ACTIVE, VALUE_FN_SOURCE
    VALUE_FN_ACTIVE = True
    VALUE_FN_SOURCE = f"value_dp:{os.path.basename(path)}"
    @torch.no_grad()
    def vf(s, player:int):
        x = torch.from_numpy(encode_features(s, player)).float().to(device)
        if x.ndim == 1: x = x.unsqueeze(0)
        y = model(x)
        return float(y.item())
    return vf

def train_value(data_path:str, save_path:str="value_single.pt", epochs:int=5, batch:int=512, lr:float=3e-4):
    import torch, numpy as np
    data = np.load(data_path)
    X = torch.from_numpy(data["x"]).float()
    Y = torch.from_numpy(data["y"]).float()
    in_dim = X.shape[1]
    params = init_value_params(in_dim)
    N = X.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            xb = X[idx]; yb = Y[idx]
            pred = value_forward(xb.numpy(), params)
            loss = ((pred - yb)**2).mean()
            for t in params.values():
                if t.grad is not None: t.grad.zero_()
            loss.backward()
            # simple SGD
            for k,t in params.items():
                with torch.no_grad():
                    t -= lr * t.grad
            tot += float(loss.item()) * xb.size(0)
        print(f"Epoch {ep+1}: loss={tot/N:.4f}")
    torch.save({k:v.detach().cpu() for k,v in params.items()}, save_path)
    print(f"Saved value params to {save_path}")

def load_value_params(path:str):
    import torch
    d = torch.load(path, map_location="cpu")
    for t in d.values():
        t.requires_grad = False
    global VALUE_FN_ACTIVE, VALUE_FN_SOURCE
    VALUE_FN_ACTIVE = True
    VALUE_FN_SOURCE = f"value_params:{os.path.basename(path)}"
    return d

# -------- MCCFR (functional) --------
def rt_get_strategy(tab:Dict, infoset:str, actions:List[Tuple], tau:float=0.0):
    # Regret matching / CFR+ baseline probabilities
    r = np.array([tab["regret"][infoset][a] for a in actions], dtype=float)
    if CFR_MODE == "CFR_PLUS":
        r = np.maximum(r, 0.0)
    if r.sum() <= 1e-12:
        # bias towards call/check when no info
        base = np.ones(len(actions))
        for i,a in enumerate(actions):
            if a[0] == 'F': base[i] = 0.5
            if a[0] in ('C','X'): base[i] = 1.5
        probs = base / base.sum()
    else:
        probs = r / r.sum()
    if tau > 0:
        probs = np.exp(np.log(np.clip(probs,1e-12,1))/(1+tau)); probs/=probs.sum()
    return {a: float(p) for a,p in zip(actions, probs.tolist())}

def rt_add_regret(tab, infoset, action, delta):
    # accumulate in float64 for stability
    prev = float(tab["regret"][infoset][action])
    new_val = float(prev + float(delta))
    if CFR_MODE == "CFR_PLUS":
        new_val = max(0.0, new_val)
    # floor to avoid extreme negatives; helps with 95/5 pruning stability
    if new_val < REGRET_FLOOR:
        new_val = REGRET_FLOOR
    tab["regret"][infoset][action] = new_val

def rt_add_strategy(tab, infoset, sigma:Dict[Tuple,float], weight:float=1.0, iteration:int=1):
    # LCFR/DCFR weighting support
    if CFR_MODE == "LCFR":
        w = float(iteration)
    elif CFR_MODE == "DCFR":
        w = float(iteration ** CFR_ALPHA)
    else:
        w = max(0.0, float(weight))
    for a,p in sigma.items():
        prev = float(tab["strategy_sum"][infoset][a])
        tab["strategy_sum"][infoset][a] = float(prev + w * float(p))

def rt_avg_strategy(tab, infoset, actions):
    denom = sum(tab["strategy_sum"][infoset][a] for a in actions) + 1e-12
    if denom <= 1e-10:
        return {a: 1.0/len(actions) for a in actions}
    return {a: tab["strategy_sum"][infoset][a]/denom for a in actions}

def info_key(s, player:int):
    if USE_ABSTRACTION:
        return abstraction_key(s, player)
    return f"{public_state_key(s)}|priv:{private_obs(s, player)}|p:{player}"

def terminal_value(s, target:int):
    return utility(s, target)

def rollout_value(s, target:int, value_fn=None):
    if value_fn is not None:
        try:
            return float(value_fn(s, target))
        except Exception:
            pass
    ss = clone_state(s)
    steps = 0
    while not is_terminal(ss) and steps < 1000:
        if is_chance_node(ss):
            ss = chance_step(ss); continue
        la = legal_actions(ss)
        if not la: steps += 1; continue
        a = random.choice(la); ss = step(ss, a); steps += 1
    return terminal_value(ss, target)

def external_sampling(tab, s, target:int, reach_i:float, reach_opp:float, depth:int, value_fn, depth_limit, iteration:int, linear_weighting:bool=True):
    if is_chance_node(s):
        return external_sampling(tab, chance_step(s), target, reach_i, reach_opp, depth, value_fn, depth_limit, iteration, linear_weighting)
    if is_terminal(s):
        return terminal_value(s, target)

    player = s["to_act"]
    legal = legal_actions(s)

    # If no legal actions (e.g., all-in freeze), fallback to rollout to advance state
    if not legal:
        return rollout_value(s, target, value_fn=value_fn)

    if depth_limit is not None and depth >= depth_limit:
        # At depth cap, use value function if available (leaf evaluation)
        return rollout_value(s, target, value_fn=value_fn)

    infoset = info_key(s, player)
    sigma = rt_get_strategy(tab, infoset, legal) if legal else {}
    # Variance reduction: simple baseline control variate per node
    baseline_val = 0.0
    if VR_MCCFR and legal:
        try:
            # Expected value under current sigma as baseline
            acts_list = list(legal)
            probs_list = [float(max(sigma.get(a, 0.0), 0.0)) for a in acts_list]
            zprob = sum(probs_list)
            if zprob <= 1e-12:
                probs_list = [1.0/len(acts_list)] * len(acts_list)
            else:
                probs_list = [p/zprob for p in probs_list]
            ev = 0.0
            for a, p_a in zip(acts_list, probs_list):
                try:
                    ev += p_a * rollout_value(step(s, a), target, value_fn=value_fn)
                except Exception:
                    continue
            baseline_val = float(ev)
        except Exception:
            baseline_val = 0.0

    if player == target:
        # Build action list and probs robustly
        acts = list(legal)
        if not acts:
            return rollout_value(s, target, value_fn=value_fn)
        probs = [max(float(sigma.get(a, 0.0)), 0.0) for a in acts]
        z = sum(probs)
        if z <= 1e-12:
            probs = [1.0/len(acts)] * len(acts)
            z = 1.0

        # Negative-regret pruning (skip strongly bad actions in 95% iters; keep all in 5%)
        if NEG_PRUNE_ACTIVE and stage(s) not in NEG_PRUNE_STAGE_DISABLE and random.random() < NEG_PRUNE_PROB:
            # keep actions with regret above threshold; ensure at least one action stays
            kept = []
            regrets = [float(tab["regret"][infoset][a]) for a in acts]
            for a, r in zip(acts, regrets):
                if r >= NEG_PRUNE_THRESHOLD:
                    kept.append(a)
            if not kept:
                # keep the best (least negative) action to avoid empty set
                best_idx = int(np.argmax(regrets))
                kept = [acts[best_idx]]
            # re-normalize to kept set
            kp = set(kept)
            acts = [a for a in acts if a in kp]
            probs = [1.0/len(acts)] * len(acts)

        util = {}
        node_util = 0.0
        for a, p in zip(acts, probs):
            nxt = step(s, a)
            u = external_sampling(tab, nxt, target, reach_i*p, reach_opp, depth+1, value_fn, depth_limit, iteration, linear_weighting)
            if VR_MCCFR:
                util[a] = u - baseline_val
                node_util += p * util[a]
            else:
                util[a] = u
                node_util += p * u
        # Regret update (CFR+ with iteration-weighted regrets)
        for a in acts:
            regret = util[a] - node_util
            w = iteration if linear_weighting else 1.0
            rt_add_regret(tab, infoset, a, w * regret * reach_opp)
        # Strategy sum update with normalized probs, weighted by reach_i
        sig = {a: (p if z == 1.0 else p/z) for a, p in zip(acts, probs)}
        rt_add_strategy(tab, infoset, sig, weight=reach_i, iteration=iteration)
        return node_util
    else:
        # Opponent sampling: use sigma if valid else uniform over legal
        acts = list(legal)
        if not acts:
            return rollout_value(s, target, value_fn=value_fn)
        # Public Chance Sampling variant: reweight by simple baseline to reduce variance
        probs = [max(float(sigma.get(a, 0.0)), 0.0) for a in acts]
        z = sum(probs)
        if z <= 1e-12:
            probs = [1.0/len(acts)] * len(acts)
        a = random.choices(acts, weights=probs, k=1)[0]
        nxt = step(s, a)
        p = sigma.get(a, 1.0/len(acts))
        return external_sampling(tab, nxt, target, reach_i, reach_opp * p, depth+1, value_fn, depth_limit, iteration, linear_weighting)

def mccfr_train_iteration(tab, root, value_fn=None, depth_limit=None, iteration:int=1, linear_weighting=True):
    for p in range(root["num_players"]):
        external_sampling(tab, root, p, 1.0, 1.0, 0, value_fn, depth_limit, iteration, linear_weighting)

def depth_limited_resolve(root, value_fn, cfr_iters:int=200, depth_limit:int=3, blueprint:Optional[Dict]=None):
    # Respect caller values; do not override upwards to global caps during strict eval
    cfr_iters = int(max(1, cfr_iters))
    depth_limit = int(max(1, depth_limit))

    def _warmstart_from_blueprint(tab, s):
        if not blueprint or RESOLVE_WARMSTART_WEIGHT <= 0:
            return
        legal = legal_actions(s)
        if not legal:
            return
        infoset = info_key(s, s["to_act"]) if not USE_ABSTRACTION else abstraction_key(s, s["to_act"])  # consistent key
        if infoset in blueprint:
            options = []
            probs = []
            for k, v in blueprint[infoset].items():
                ks = k.replace('[', '(').replace(']', ')').replace('"', '').replace("'", '')
                if ks.startswith("(R,"):
                    num = ''.join(ch for ch in ks if ch.isdigit())
                    a = ('R', int(num))
                elif ks.startswith("(C"):
                    a = ('C',)
                elif ks.startswith("(X"):
                    a = ('X',)
                elif ks.startswith("(F"):
                    a = ('F',)
                else:
                    continue
                if a in legal:
                    options.append(a); probs.append(float(v))
            z = sum(probs)
            if z > 0 and options:
                sig = {a: (p / z) for a, p in zip(options, probs)}
                # seed strategy_sum at root
                rt_add_strategy(tab, info_key(s, s["to_act"]), sig, weight=float(RESOLVE_WARMSTART_WEIGHT), iteration=1)

    def _resolve_once(state, iters:int) -> Dict[Tuple, float]:
        tab = {"regret": defaultdict(lambda: defaultdict(float)),
               "strategy_sum": defaultdict(lambda: defaultdict(float))}
        _warmstart_from_blueprint(tab, state)
        for it in range(1, iters+1):
            mccfr_train_iteration(tab, state, value_fn=value_fn, depth_limit=depth_limit, iteration=it, linear_weighting=True)
        infoset = info_key(state, state["to_act"])
        legal = legal_actions(state)
        base = rt_avg_strategy(tab, infoset, legal) if legal else {}
        if legal and blueprint and KL_ALPHA and KL_ALPHA > 0.0:
            # Blend with blueprint prior at root (simple convex combination)
            prior = {}
            bp_key = infoset if not USE_ABSTRACTION else abstraction_key(state, state["to_act"])  # consistent lookup
            if bp_key in blueprint:
                for k, v in blueprint[bp_key].items():
                    ks = k.replace('[', '(').replace(']', ')').replace('"', '').replace("'", '')
                    if ks.startswith("(R,"):
                        num = ''.join(ch for ch in ks if ch.isdigit()); a = ('R', int(num))
                    elif ks.startswith("(C"):
                        a = ('C',)
                    elif ks.startswith("(X"):
                        a = ('X',)
                    elif ks.startswith("(F"):
                        a = ('F',)
                    else:
                        continue
                    if a in legal:
                        prior[a] = prior.get(a, 0.0) + float(v)
                z = sum(prior.values())
                if z > 0:
                    for a in list(prior.keys()):
                        prior[a] /= z
            if prior:
                # linear blend: sigma = (1-KL_ALPHA)*base + KL_ALPHA*prior (renormalize to legal)
                out = {}
                for a in legal:
                    out[a] = (1.0 - KL_ALPHA) * float(base.get(a, 0.0)) + KL_ALPHA * float(prior.get(a, 0.0))
                z2 = sum(out.values())
                if z2 > 0:
                    for a in list(out.keys()):
                        out[a] /= z2
                return out
        return base

    # Small cache by public state key + to_act + resolve params/modes
    global RESOLVE_CACHE
    # Expand cache key with action ladder flags and coarse public features
    ps = float(pot_size(root))
    me = root["players"][root["to_act"]]
    to_call2 = max(0, root["current_bet"] - me["invested"]) 
    pot_after_call = ps + to_call2
    spr = 0.0 if pot_after_call <= 0 else me["stack"] / pot_after_call
    key = (
        public_state_key(root),
        root["to_act"],
        cfr_iters,
        depth_limit,
        int(bool(blueprint)),
        RESOLVE_DETERMS,
        RESOLVE_WARMSTART_WEIGHT,
        HOLE_EVAL_SAMPLES,
        float(KL_ALPHA),
        int(bool(RESOLVE_PARALLEL)),
        int(bool(POSTERIOR_FROM_BLUEPRINT)),
        int(bool(ACTION_ENSURE_POT_NODE)),
        int(bool(ACTION_ENSURE_JAM_NODE)),
        round(spr, 2),
    )
    if key in RESOLVE_CACHE:
        return RESOLVE_CACHE[key]

    # Determinization loop (public belief)
    if RESOLVE_DETERMS and RESOLVE_DETERMS > 1:
        agg: Dict[Tuple, float] = {}
        per_iter = max(1, cfr_iters // RESOLVE_DETERMS)
        if RESOLVE_PARALLEL and RESOLVE_DETERMS >= 2:
            from multiprocessing import get_context
            ctx = get_context("spawn")
            def _job(_):
                ds = _sample_determinization(root, root["to_act"], blueprint)
                return _resolve_once(ds, per_iter)
            with ctx.Pool(processes=min(RESOLVE_DETERMS, max(1, os.cpu_count() or 1))) as pool:
                for strat in pool.map(_job, list(range(RESOLVE_DETERMS))):
                    for a, p in strat.items():
                        agg[a] = agg.get(a, 0.0) + float(p)
        else:
            for _ in range(RESOLVE_DETERMS):
                ds = _sample_determinization(root, root["to_act"], blueprint)
                strat = _resolve_once(ds, per_iter)
                for a, p in strat.items():
                    agg[a] = agg.get(a, 0.0) + float(p)
        z = sum(agg.values())
        strat = {a: (v / z) for a, v in agg.items()} if z > 0 else {}
    else:
        strat = _resolve_once(root, cfr_iters)

    if len(RESOLVE_CACHE) > RESOLVE_CACHE_MAX:
        RESOLVE_CACHE.clear()
    RESOLVE_CACHE[key] = strat
    return strat

def _sample_determinization(s, me_idx:int, blueprint: Optional[Dict]=None):
    # Clone state and fill unknown opponents' hole cards.
    # If POSTERIOR_FROM_BLUEPRINT: bias sampling by blueprint posterior given public state; else uniform.
    ss = clone_state(s)
    public_key = public_state_key(ss)
    # Build remaining deck
    known = set(ss["board"]) | set(ss["players"][me_idx]["hole"]) | set()
    for i, p in enumerate(ss["players"]):
        if i == me_idx:
            continue
        for c in p.get("hole", []):
            known.add(c)
    remaining = [c for c in range(52) if c not in known]
    random.shuffle(remaining)

    # Helper: posterior over pairs (c1,c2) for a seat
    def _posterior_pairs_for_seat(seat:int) -> List[Tuple[int,int,float]]:
        # Uniform fallback
        pairs = []
        m = {}
        n = len(remaining)
        # Very light blueprint-based posterior: prefer strong hole buckets on current board
        if not POSTERIOR_FROM_BLUEPRINT or not blueprint:
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((remaining[i], remaining[j], 1.0))
            return pairs
        # Score by treys rank on current board
        try:
            b = board_to_treys(ss["board"]) if ss["board"] else []
            for i in range(n):
                for j in range(i+1, n):
                    h = hand_to_treys([remaining[i], remaining[j]])
                    score = EVAL.evaluate(b, h)
                    strength = (7462 - score) / 7462.0
                    pairs.append((remaining[i], remaining[j], 0.2 + 0.8*strength))
        except Exception:
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((remaining[i], remaining[j], 1.0))
        return pairs

    # Assign to opponents sequentially without overlap
    used = set()
    for i, p in enumerate(ss["players"]):
        if i == me_idx or p.get("folded", False):
            continue
        if len(p.get("hole", [])) >= 2:
            continue
        cand = _posterior_pairs_for_seat(i)
        # Filter out already used or conflicting cards
        cand = [(c1, c2, w) for (c1, c2, w) in cand if (c1 not in used) and (c2 not in used)]
        if not cand:
            continue
        # Sample pair by weights
        weights = [w for (_, _, w) in cand]
        k = random.choices(range(len(cand)), weights=weights, k=1)[0]
        c1, c2, _ = cand[k]
        p.setdefault("hole", []).extend([c1, c2])
        used.add(c1); used.add(c2)
    return ss

# -------- Data / Scripts --------
def rollout_return(s, player:int) -> float:
    steps=0; ss=clone_state(s)
    while not is_terminal(ss) and steps < 1000:
        if is_chance_node(ss): ss = chance_step(ss); continue
        la = legal_actions(ss)
        if not la: steps += 1; continue
        raises = [a for a in la if a[0]=='R']
        if raises and random.random()<0.15: a=random.choice(raises)
        else: a=('C',) if ('C',) in la else ('X',) if ('X',) in la else random.choice(la)
        ss = step(ss,a); steps+=1
    return utility(ss, player)

def cmd_selfplay_generate(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    xs=[]; ys=[]; btn=0
    for _ in trange(args.episodes, desc="Self-play"):
        s = new_hand(btn=btn, num_players=args.num_players)
        for seat in range(s["num_players"]):
            ss = clone_state(s)
            for _ in range(random.randint(0,6)):
                if is_chance_node(ss): ss = chance_step(ss)
                else:
                    la = legal_actions(ss)
                    if not la: break
                    ss = step(ss, random.choice(la))
            xs.append(encode_features(ss, seat))
            ys.append(rollout_return(ss, seat))
        btn = (btn + 1) % s["num_players"]
    xs=np.stack(xs,axis=0); ys=np.asarray(ys,dtype=np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, x=xs, y=ys)
    print(f"Saved self-play dataset to {args.out}  x{xs.shape} y{ys.shape}")

def cmd_generate_cfv(args):
    # Generate counterfactual value targets using strict BR gap on sampled states
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    bp = load_blueprint(args.blueprint) if args.blueprint else {}
    if args.value and os.path.exists(args.value):
        value_fn = make_value_fn_from_dp(args.value) if args.value.endswith('.pt') else make_value_fn(load_value_params(args.value))
    else:
        value_fn = (lambda s, p: 0.0)
    if VALUE_FN_ACTIVE:
        try:
            print(json.dumps({"value_fn": {"active": True, "source": VALUE_FN_SOURCE}}))
        except Exception:
            pass
    set_resolve_enhancements(getattr(args, 'resolve_determs', None), getattr(args, 'resolve_warmstart', None), getattr(args, 'hole_eval_samples', None))
    set_resolve_regularization(getattr(args, 'kl_alpha', None))
    X = []
    Y = []
    for _ in trange(args.samples, desc="CFV-generate"):
        s = _sample_state(args.num_players)
        player = random.randrange(args.num_players)
        try:
            memo = {}
            br = _br_value(s, player, bp, value_fn, max(RESOLVE_ITERS, args.resolve_iters), max(RESOLVE_DEPTH, args.resolve_depth), args.depth_cap, memo)
            sv = _sigma_value(s, player, bp, value_fn, max(RESOLVE_ITERS, args.resolve_iters), max(RESOLVE_DEPTH, args.resolve_depth), args.depth_cap, memo)
            gap = max(0.0, br - sv)
            X.append(encode_features(s, player))
            Y.append(gap)
        except Exception:
            continue
    if X:
        Xn = np.stack(X, axis=0)
        Yn = np.asarray(Y, dtype=np.float32)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(args.out, x=Xn, y=Yn)
        print(f"Saved CFV dataset to {args.out}  x{Xn.shape} y{Yn.shape}")
    else:
        print("No CFV samples generated.")

def cmd_train_value(args):
    train_value(args.data, save_path=args.save, epochs=args.epochs, batch=args.batch, lr=args.lr)

def cmd_train_blueprint(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    global USE_ABSTRACTION, DYN_ABS_ENABLED
    if args.no_abstraction:
        pass
    else:
        if args.use_abstraction or maybe_autoload_abstraction(args.abstraction):
            USE_ABSTRACTION = True
            DYN_ABS_ENABLED = bool(getattr(args, "dynamic_abstraction", True))
    tab = {"regret": defaultdict(lambda: defaultdict(float)),
           "strategy_sum": defaultdict(lambda: defaultdict(float))}
    btn=0
    decay = float(args.regret_decay)
    for it in trange(args.iterations, desc="MCCFR"):
        s = new_hand(btn=btn, num_players=args.num_players)
        mccfr_train_iteration(tab, s, value_fn=None, depth_limit=None, iteration=it, linear_weighting=True)
        # LCFR/DCFR slice discount: каждые --slice итераций "усаживаем" регреты и суммы
        if it % max(1, int(args.slice or 10000)) == 0:
            T = it // max(1, int(args.slice or 10000))
            disc = T / float(T + 1)
            for infoset, acts in list(tab["regret"].items()):
                for a in list(acts.keys()):
                    tab["regret"][infoset][a] = float(max(REGRET_FLOOR, tab["regret"][infoset][a] * disc))
            for infoset, acts in list(tab["strategy_sum"].items()):
                for a in list(acts.keys()):
                    tab["strategy_sum"][infoset][a] = float(tab["strategy_sum"][infoset][a] * disc)
        elif decay > 0:
            # simple exponential decay of regrets
            for infoset, acts in list(tab["regret"].items()):
                for a in list(acts.keys()):
                    tab["regret"][infoset][a] *= (1.0 - decay)
        btn = (btn + 1) % s["num_players"]
    # avg strategy dump
    strategy = {}
    for infoset, acts in tab["strategy_sum"].items():
        denom = sum(acts.values()) + 1e-12
        strategy[infoset] = {str(a): float(v/denom) for a,v in acts.items()}
    with open(args.save, "w") as f: json.dump(strategy, f)
    print(f"Saved blueprint to {args.save}")
    if args.save_sum:
        raw = {}
        for infoset, acts in tab["strategy_sum"].items():
            raw[infoset] = {str(a): float(v) for a,v in acts.items()}
        with open(args.save_sum,"w") as f: json.dump(raw,f)
        print(f"Saved raw strategy_sum to {args.save_sum}")

def _merge_strategy_sums(dst:Dict, src:Dict):
    for infoset, acts in src.items():
        if infoset not in dst:
            dst[infoset] = {}
        for a, v in acts.items():
            dst[infoset][a] = dst[infoset].get(a, 0.0) + float(v)

def _worker_mccfr(args_tuple):
    iterations, num_players, seed, bet_pre, bet_post, abstraction_path, use_abs, dyn_abs = args_tuple
    random.seed(seed)
    if bet_pre: set_bet_sizes(bet_pre, None)
    if bet_post: set_bet_sizes(None, bet_post)
    if abstraction_path:
        try:
            load_abstraction(abstraction_path)
        except Exception:
            pass
    if use_abs:
        global USE_ABSTRACTION, DYN_ABS_ENABLED
        USE_ABSTRACTION = True
        DYN_ABS_ENABLED = bool(dyn_abs)
    tab = {"regret": defaultdict(lambda: defaultdict(float)),
           "strategy_sum": defaultdict(lambda: defaultdict(float))}
    btn = 0
    for it in range(1, iterations+1):
        s = new_hand(btn=btn, num_players=num_players)
        mccfr_train_iteration(tab, s, value_fn=None, depth_limit=None, iteration=it, linear_weighting=True)
        btn = (btn + 1) % num_players
    # Serialize strategy_sum only
    out = {}
    for infoset, acts in tab["strategy_sum"].items():
        out[infoset] = {str(a): float(v) for a, v in acts.items()}
    return out

def cmd_train_blueprint_dist(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    # Apply resolve enhancements globally for periodic h2h/eval
    set_resolve_enhancements(getattr(args, 'resolve_determs', None), getattr(args, 'resolve_warmstart', None), getattr(args, 'hole_eval_samples', None))
    world = max(1, int(args.num_workers))
    total_iters = int(args.iterations)
    chunk = int(args.chunk) if args.chunk else max(1, total_iters // world)
    seeds = [1234 + i for i in range(world)]
    bet_pre = args.bet_sizes_preflop
    bet_post = args.bet_sizes_postflop
    abstraction_path = args.abstraction
    use_abs = bool(args.use_abstraction)
    dyn_abs = bool(args.dynamic_abstraction)
    print(f"[bold green]Distributed MCCFR[/bold green]: workers={world} total_iters={total_iters} chunk={chunk}")
    try:
        print(json.dumps({
            "config": {
                "iterations": total_iters,
                "num_workers": world,
                "chunk": chunk,
                "raise_cap": RAISE_CAP,
                "resolve": {
                    "iters": RESOLVE_ITERS,
                    "depth": RESOLVE_DEPTH,
                    "determs": RESOLVE_DETERMS,
                    "warmstart": RESOLVE_WARMSTART_WEIGHT,
                    "kl_alpha": KL_ALPHA,
                    "parallel": RESOLVE_PARALLEL,
                    "posterior": POSTERIOR_FROM_BLUEPRINT
                },
                "abstraction": {
                    "use": bool(use_abs),
                    "dynamic": bool(dyn_abs),
                    "path": abstraction_path
                }
            }
        }))
    except Exception:
        pass
    # Allow toggling posterior-based determinization and parallel resolve via CLI
    set_resolve_modes(getattr(args, 'resolve_parallel', None), getattr(args, 'posterior_from_blueprint', None))
    # Apply pruning parameters via CLI
    set_pruning_params(getattr(args, 'neg_prune_prob', None), getattr(args, 'neg_prune_threshold', None), getattr(args, 'regret_floor', None))
    merged = {}
    done = 0
    # optional checkpoint auto-clean helpers
    def _dir_size_bytes(path:str) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
        return total

    def _maybe_autoclean_ckpts(dir_path:str):
        try:
            keep_every = int(getattr(args, 'ckpt_keep_every', 0) or 0)
            max_gb = float(getattr(args, 'ckpt_max_gb', 0.0) or 0.0)
            if (keep_every <= 0 and max_gb <= 0.0) or not os.path.isdir(dir_path):
                return
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.startswith('ckpt_') and f.endswith('.json')]
            files.sort(key=lambda p: (int(''.join(ch for ch in os.path.basename(p) if ch.isdigit()) or '0')))
            # Keep every Nth, delete others except last
            if keep_every > 0:
                keep = set()
                for pth in files:
                    try:
                        it = int(''.join(ch for ch in os.path.basename(pth) if ch.isdigit()) or '0')
                        if it % keep_every == 0:
                            keep.add(pth)
                    except Exception:
                        pass
                # Always keep the latest
                if files:
                    keep.add(files[-1])
                for pth in files:
                    if pth not in keep:
                        try:
                            os.remove(pth)
                        except Exception:
                            pass
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.startswith('ckpt_') and f.endswith('.json')]
                files.sort(key=lambda p: (int(''.join(ch for ch in os.path.basename(p) if ch.isdigit()) or '0')))
            # Enforce max size in GB
            if max_gb > 0.0:
                limit = max_gb * (1024**3)
                # delete oldest until under limit
                while _dir_size_bytes(dir_path) > limit and len(files) > 1:
                    victim = files.pop(0)
                    try:
                        os.remove(victim)
                    except Exception:
                        pass
        except Exception:
            pass
    def _maybe_autoclean_logs(dir_path:str):
        try:
            if not dir_path or not os.path.isdir(dir_path):
                return
            keep_days = int(getattr(args, 'logs_keep_days', 0) or 0)
            max_gb = float(getattr(args, 'logs_max_gb', 0.0) or 0.0)
            now = time.time()
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.log') or f.endswith('.jsonl')]
            files.sort(key=lambda p: os.path.getmtime(p))
            # age pruning
            if keep_days > 0:
                cutoff = now - keep_days * 86400
                for p in files:
                    try:
                        if os.path.getmtime(p) < cutoff:
                            os.remove(p)
                    except Exception:
                        pass
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.log') or f.endswith('.jsonl')]
                files.sort(key=lambda p: os.path.getmtime(p))
            # size pruning
            if max_gb > 0.0:
                limit = max_gb * (1024**3)
                def _size(paths):
                    s=0
                    for pp in paths:
                        try: s+=os.path.getsize(pp)
                        except Exception: pass
                    return s
                while _size(files) > limit and len(files) > 1:
                    victim = files.pop(0)
                    try: os.remove(victim)
                    except Exception: pass
        except Exception:
            pass
    def _quick_eval_path(bp_path:str):
        try:
            if not args.eval_every_chunk:
                return
            bp_local = load_blueprint(bp_path)
            # value fn
            if args.eval_value and os.path.exists(args.eval_value):
                value_fn = make_value_fn_from_dp(args.eval_value) if args.eval_value.endswith('.pt') else make_value_fn(load_value_params(args.eval_value))
            else:
                value_fn = (lambda s, p: 0.0)
            # abstraction
            if args.eval_abstraction:
                try:
                    load_abstraction(args.eval_abstraction)
                except Exception:
                    pass
            # run minimal evaluate loop
            btn = 0
            winnings = []
            baseline = args.eval_baseline or 'random'
            bfn = _baseline_random_action if baseline=='random' else _baseline_tight_action
            t0 = time.time()
            for _ in range(int(args.eval_hands)):
                res, btn = _play_hand(bp_local, value_fn, bfn, num_players=int(args.eval_num_players), btn=btn)
                winnings.append(res[0])
            dt = max(1e-9, time.time()-t0)
            bb = BIG_BLIND
            bb_per_hand = [w / bb for w in winnings]
            mean_bb100 = float(np.mean(bb_per_hand) * 100)
            std = float(np.std(bb_per_hand))
            n = max(1, len(bb_per_hand))
            se = std / math.sqrt(n)
            ci_half = 1.96 * se * 100.0  # in bb/100 units
            rate_hps = len(winnings) / dt
            print(json.dumps({
                "monitor": {
                    "done_iters": done,
                    "bb_100": mean_bb100,
                    "std_bb": std,
                    "ci95_halfwidth": ci_half,
                    "hands_per_sec": rate_hps,
                    "hands": len(winnings),
                    "baseline": baseline,
                    "ckpt": bp_path
                }
            }))

            # Optional head-to-head vs provided opponent blueprint each chunk
            if getattr(args, 'h2h_every_chunk', False) and getattr(args, 'h2h_blueprint', None):
                opp_path = args.h2h_blueprint
                if isinstance(opp_path, str) and os.path.exists(opp_path):
                    try:
                        bp_opp = load_blueprint(opp_path)
                        h2h_btn = 0
                        h2h_wins = []
                        h2h_hands = int(getattr(args, 'h2h_hands', 1000) or 1000)
                        resolve_iters = int(getattr(args, 'h2h_resolve_iters', RESOLVE_ITERS) or RESOLVE_ITERS)
                        resolve_depth = int(getattr(args, 'h2h_resolve_depth', RESOLVE_DEPTH) or RESOLVE_DEPTH)
                        for _ in range(h2h_hands):
                            res, h2h_btn = _play_hand_matchup(bp_local, bp_opp, value_fn, num_players=int(args.eval_num_players), btn=h2h_btn, resolve_iters=resolve_iters, resolve_depth=resolve_depth)
                            h2h_wins.append(res[0])
                        h2h_bb_per_hand = [w / bb for w in h2h_wins]
                        h2h_bb100 = float(np.mean(h2h_bb_per_hand) * 100)
                        print(json.dumps({
                            "head2head": {
                                "done_iters": done,
                                "bb_100": h2h_bb100,
                                "hands": len(h2h_wins),
                                "ckpt": bp_path,
                                "opp": opp_path,
                                "resolve_iters": resolve_iters,
                                "resolve_depth": resolve_depth
                            }
                        }))
                    except Exception:
                        pass
        except Exception:
            pass

    with mp.get_context("spawn").Pool(processes=world) as pool:
        t_batch0 = time.time()
        while done < total_iters:
            batch_iters = min(chunk, total_iters - done)
            parts = pool.map(_worker_mccfr, [
                (batch_iters // world + (1 if i < (batch_iters % world) else 0),
                 args.num_players, seeds[i] + done, bet_pre, bet_post, abstraction_path, use_abs, dyn_abs)
                for i in range(world)
            ])
            for part in parts:
                _merge_strategy_sums(merged, part)
            # LCFR/DCFR discounted resets by "time slices" (Pluribus-style): every slice shrink regrets & sums
            slice_size = int(getattr(args, 'slice', 10000))
            if slice_size > 0 and done % slice_size == 0:
                # T is slice index
                T = done // slice_size
                disc = T / float(T + 1)
                for infoset, acts in list(merged.items()):
                    if infoset in merged:
                        for a in list(acts.keys()):
                            merged[infoset][a] = float(max(REGRET_FLOOR, merged[infoset][a] * disc))
            done += batch_iters
            elapsed = max(1e-9, time.time() - t_batch0)
            rate = done / elapsed
            eta = (total_iters - done) / max(1e-9, rate)
            print(json.dumps({"progress": {"done_iters": done, "total_iters": total_iters, "rate_it_per_s": rate, "eta_s": eta}}))
            # Optional checkpoint
            if args.checkpoint_dir:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_{done}.json")
                bp = {}
                for infoset, acts in merged.items():
                    denom = sum(acts.values()) + 1e-12
                    bp[infoset] = {a: float(v/denom) for a, v in acts.items()}
                with open(ckpt_path, "w") as f: json.dump(bp, f)
                print(f"Checkpoint saved: {ckpt_path}")
                _quick_eval_path(ckpt_path)
                _maybe_autoclean_ckpts(args.checkpoint_dir)
                if getattr(args, 'logs_dir', None):
                    os.makedirs(args.logs_dir, exist_ok=True)
                    _maybe_autoclean_logs(args.logs_dir)
                try:
                    if getattr(args, 'milestones_jsonl', None):
                        os.makedirs(os.path.dirname(args.milestones_jsonl) or ".", exist_ok=True)
                        with open(args.milestones_jsonl, "a") as mf:
                            mf.write(json.dumps({"milestone": {"done_iters": done, "ckpt": ckpt_path}}) + "\n")
                except Exception:
                    pass
                # Save milestone blueprint snapshot (normalized)
                try:
                    m_int = int(getattr(args, 'milestone_interval', 0) or 0)
                    m_dir = getattr(args, 'milestone_dir', None)
                    if m_int > 0 and m_dir and (done % m_int == 0):
                        out_dir = os.path.join(m_dir, f"milestone_{done}")
                        os.makedirs(out_dir, exist_ok=True)
                        m_path = os.path.join(out_dir, "blueprint.json")
                        with open(m_path, "w") as mf:
                            json.dump(bp, mf)
                        print(f"Milestone saved: {m_path}")
                except Exception:
                    pass
    # Normalize to blueprint
    blueprint = {}
    for infoset, acts in merged.items():
        denom = sum(acts.values()) + 1e-12
        blueprint[infoset] = {a: float(v/denom) for a, v in acts.items()}
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w") as f: json.dump(blueprint, f)
    print(f"Saved distributed blueprint to {args.save}")
    try:
        print(json.dumps({"training_summary": {"total_infosets": len(blueprint), "iterations": total_iters, "workers": world}}))
    except Exception:
        pass
    if args.save_sum:
        with open(args.save_sum, "w") as f: json.dump(merged, f)
        print(f"Saved merged raw sums to {args.save_sum}")

def cmd_merge_sums(args):
    agg = {}
    for path in args.inputs:
        with open(path,"r") as f:
            data = json.load(f)
        for infoset, acts in data.items():
            if infoset not in agg: agg[infoset]={}
            for a,v in acts.items():
                agg[infoset][a] = agg[infoset].get(a,0.0)+float(v)
    blueprint={}
    for infoset, acts in agg.items():
        denom = sum(acts.values()) + 1e-12
        blueprint[infoset] = {a: float(v/denom) for a,v in acts.items()}
    with open(args.out_blueprint,"w") as f: json.dump(blueprint,f)
    print(f"Saved merged blueprint to {args.out_blueprint}")
    if args.out_sum:
        with open(args.out_sum,"w") as f: json.dump(agg,f)
        print(f"Saved merged raw sums to {args.out_sum}")

def choose_bot_action(s, blueprint:Dict, value_fn, resolve_iters:int=None, resolve_depth:int=None):
    la = legal_actions(s)
    if not la: return None
    # Stage-aware defaults; can be overridden by args
    st = stage(s)
    it = resolve_iters if resolve_iters is not None else (200 if st=='preflop' else 400)
    dp = resolve_depth if resolve_depth is not None else (3 if st=='preflop' else 4)
    strat = depth_limited_resolve(s, value_fn=value_fn, cfr_iters=it, depth_limit=dp, blueprint=blueprint)
    if strat:
        acts=list(strat.keys()); probs=[strat[a] for a in acts]
        return random.choices(acts, weights=probs, k=1)[0]
    infoset = f"{public_state_key(s)}|priv:{private_obs(s, s['to_act'])}|p:{s['to_act']}"
    if infoset in blueprint:
        options=[]; probs=[]
        for k,v in blueprint[infoset].items():
            ks = k.replace('[','(').replace(']',')').replace('"','').replace("'", '')
            if ks.startswith("(R,"):
                num = ''.join(ch for ch in ks if ch.isdigit()); options.append(('R', int(num)))
            elif ks.startswith("(C"): options.append(('C',))
            elif ks.startswith("(X"): options.append(('X',))
            elif ks.startswith("(F"): options.append(('F',))
            else: continue
            probs.append(float(v))
        if options:
            return random.choices(options, weights=probs, k=1)[0]
    return random.choice(la)

def _baseline_random_action(s):
    la = legal_actions(s)
    if not la: return None
    raises = [a for a in la if a[0]=='R']
    if raises and random.random() < 0.1: return random.choice(raises)
    if ('C',) in la: return ('C',)
    if ('X',) in la: return ('X',)
    return random.choice(la)

def _baseline_tight_action(s):
    la = legal_actions(s)
    if not la: return None
    if ('C',) in la and random.random() < 0.6: return ('C',)
    if ('X',) in la and random.random() < 0.8: return ('X',)
    raises = [a for a in la if a[0]=='R']
    if raises and random.random() < 0.05: return random.choice(raises)
    if ('F',) in la: return ('F',)
    return random.choice(la)

def _play_hand(blueprint:Dict, value_fn, baseline_fn, num_players:int=6, btn:int=0):
    s = new_hand(btn=btn, num_players=num_players)
    hero_stack_start = s["players"][0]["stack"]  # Track starting stack for hero (seat 0)
    while True:
        while is_chance_node(s): s = chance_step(s)
        if is_terminal(s):
            delta_chips = s["players"][0]["stack"] - hero_stack_start
            return [delta_chips for _ in range(num_players)], (btn + 1) % num_players
        # If no legal actions (e.g., all-in freeze), auto-advance state safely
        la0 = legal_actions(s)
        if not la0:
            active = [i for i, pl in enumerate(s["players"]) if not pl["folded"]]
            if len(active) <= 1:
                _award_by_fold(s)
                continue
            someone_to_call = any((s["players"][i]["invested"] < s["current_bet"] and not s["players"][i]["all_in"]) for i in active)
            if not someone_to_call:
                _end_bet_round(s)
                continue
            # Move action to next eligible player
            n = s["num_players"]
            nxt = (s["to_act"] + 1) % n
            while (s["players"][nxt]["folded"] or s["players"][nxt]["all_in"]) and nxt != s["to_act"]:
                nxt = (nxt + 1) % n
            s["to_act"] = nxt
            continue
        cp = s["to_act"]
        if cp == 0:
            a = choose_bot_action(s, blueprint, value_fn)
        else:
            a = baseline_fn(s)
        if a is None:
            # Fallback to a safe legal action
            la = legal_actions(s)
            if not la:
                continue
            if ('C',) in la:
                a = ('C',)
            elif ('X',) in la:
                a = ('X',)
            else:
                a = random.choice(la)
        s = step(s, a)

def _play_hand_matchup(bp0:Dict, bp1:Dict, value_fn, num_players:int=2, btn:int=0, resolve_iters:Optional[int]=None, resolve_depth:Optional[int]=None):
    s = new_hand(btn=btn, num_players=num_players)
    hero_stack_start = s["players"][0]["stack"]  # Track starting stack for hero (seat 0)
    while True:
        while is_chance_node(s): s = chance_step(s)
        if is_terminal(s):
            delta_chips = s["players"][0]["stack"] - hero_stack_start
            return [delta_chips for _ in range(num_players)], (btn + 1) % num_players
        la0 = legal_actions(s)
        if not la0:
            active = [i for i, pl in enumerate(s["players"]) if not pl["folded"]]
            if len(active) <= 1:
                _award_by_fold(s)
                continue
            someone_to_call = any((s["players"][i]["invested"] < s["current_bet"] and not s["players"][i]["all_in"]) for i in active)
            if not someone_to_call:
                _end_bet_round(s)
                continue
            n = s["num_players"]
            nxt = (s["to_act"] + 1) % n
            while (s["players"][nxt]["folded"] or s["players"][nxt]["all_in"]) and nxt != s["to_act"]:
                nxt = (nxt + 1) % n
            s["to_act"] = nxt
            continue
        cp = s["to_act"]
        if cp == 0:
            a = choose_bot_action(s, bp0, value_fn, resolve_iters=resolve_iters, resolve_depth=resolve_depth)
        elif cp == 1:
            a = choose_bot_action(s, bp1, value_fn, resolve_iters=resolve_iters, resolve_depth=resolve_depth)
        else:
            a = _baseline_random_action(s)
        if a is None:
            la = legal_actions(s)
            if not la:
                continue
            if ('C',) in la:
                a = ('C',)
            elif ('X',) in la:
                a = ('X',)
            else:
                a = random.choice(la)
        s = step(s, a)

def cmd_evaluate(args):
    if args.no_abstraction:
        pass
    else:
        if args.use_abstraction or maybe_autoload_abstraction(args.abstraction):
            pass
    # --- evaluation with optional street-root resolve & freeze-self ---
    results = []          # hero payoffs in chips
    results_aivat = []    # AIVAT-lite corrected payoffs in chips
    hands = int(args.hands or 1000)
    btn = 0
    street_root_state = None
    freeze = FreezeBook()
    hero_seat = int(getattr(args, "hero", 0))
    
    for h in range(hands):
        s = new_hand(btn=btn, num_players=args.num_players)
        hero_stack_start = s["players"][hero_seat]["stack"]  # Track starting stack
        street_root_state = snapshot_state(s)  # start of preflop
        freeze.clear()
        prev = None
        # Decisions on trajectory (for AIVAT-lite)
        traj: list[DecRec] = []
        while True:
            if is_new_street(prev, s):
                # new street begins: reset street root & clear hero freezes
                street_root_state = snapshot_state(s)
                freeze.clear()
            player = s["to_act"]
            legal = legal_actions(s)
            if not legal:
                delta_chips = s["players"][hero_seat]["stack"] - hero_stack_start
                results.append(delta_chips)
                # --- AIVAT-lite correction ---
                if AIVAT_LITE_ENABLED:
                    # baseline: use blueprint as control for all players' stochasticity
                    def bp_sigma_lookup(st, p):
                        return get_blueprint_sigma(args.blueprint, st, p)
                    corr = 0.0
                    import time
                    start_time = time.time()
                    try:
                        for rec in traj:
                            # корректируем решения всех игроков, снижая их вклад в дисперсию
                            # оценка ведётся по ценности для hero_seat
                            elapsed = time.time() - start_time
                            if elapsed > AIVAT_TIMEOUT:
                                print(f"⚠️  AIVAT-lite timeout after {elapsed:.1f}s, skipping remaining corrections")
                                break
                            q_sa = estimate_q_bp(rec.state, hero_seat, rec.action, bp_sigma_lookup,
                                                 samples=AIVAT_SAMPLES, depth=AIVAT_DEPTH)
                            q_exp = expected_q_bp(rec.state, hero_seat, bp_sigma_lookup,
                                                  samples=max(2, AIVAT_SAMPLES//2), depth=AIVAT_DEPTH)
                            corr += (q_sa - q_exp)
                        results_aivat.append(float(delta_chips) - float(corr))
                    except Exception as e:
                        # не рвём матч-оценку в случае ошибки
                        print(f"⚠️  AIVAT-lite error: {e}, using raw value")
                        results_aivat.append(float(delta_chips))
                break
            # choose action:
            if player == hero_seat:
                # build resolve subgame from street root, freezing only hero's earlier choices
                if RESOLVE_FROM_STREET_ROOT and args.resolve_iters > 0:
                    # define blueprint sigma lookup for continuation policies
                    def bp_sigma_lookup(st, p):
                        return get_blueprint_sigma(args.blueprint, st, p)  # expected in your codebase
                    # run resolve search producing local policy pi_loc for current state
                    pi_loc = resolve_subgame_from_root(
                        street_root_state, current_state=s,
                        hero=player,
                        freeze_book=freeze if FREEZE_SELF_ONLY else None,
                        iters=args.resolve_iters,
                        depth=args.resolve_depth,
                        determs=args.resolve_determs,
                        warmstart=args.resolve_warmstart,
                        kl_alpha=args.kl_alpha,
                        leaf_value_fn=lambda st: resolve_leaf_value(st, player, bp_sigma_lookup),
                        bp_sigma_lookup=bp_sigma_lookup
                    )
                    # sample action from local resolved policy at s (fallback to blueprint if absent)
                    sigma_here = pi_loc.get(policy_key(s, player), get_blueprint_sigma(args.blueprint, s, player))
                else:
                    sigma_here = get_blueprint_sigma(args.blueprint, s, player)
                # pick action and mark freeze point for hero
                a = random.choices(list(legal), weights=[max(1e-12, sigma_here.get(x, 0.0)) for x in legal], k=1)[0]
                # remember hero already acted in this infoset on this street
                freeze.mark(infoset_key_from_state(s, player))
            else:
                # opponent uses blueprint (или baseline, если задан)
                sigma_opp = opponent_policy(args, s, player)  # your existing baseline/blueprint chooser
                a = random.choices(list(legal), weights=[max(1e-12, sigma_opp.get(x, 0.0)) for x in legal], k=1)[0]
            # record decision for AIVAT-lite
            if AIVAT_LITE_ENABLED:
                traj.append(DecRec(state=deepcopy(s), player=player, action=a))
            prev = s
            s = step(s, a)
        btn = (btn + 1) % s["num_players"]
        
        # Debug print for first few hands
        if h < 10:  # Print first 10 hands for sanity check
            print(f"Hand {h}: delta_chips={results[-1]}, delta_bb={results[-1]/BIG_BLIND:.2f}")
    
    # aggregate & print (reuse existing reporter) + AIVAT-lite line
    report_results(results, BIG_BLIND, "raw")
    if AIVAT_LITE_ENABLED and results_aivat:
        report_results(results_aivat, BIG_BLIND, "AIVAT-lite")
    try:
        from __main__ import _BP_HITS, _BP_MISSES
        total = _BP_HITS + _BP_MISSES
        if total:
            print(f"[BP] lookup: hits={_BP_HITS} misses={_BP_MISSES} hit_rate={100.0*_BP_HITS/total:.1f}%")
    except Exception:
        pass
    return 0
  try:
    print(f"[BP] lookup: hits={_BP_HITS} misses={_BP_MISSES} hit_rate={100.0*_BP_HITS/max(1,_BP_HITS+_BP_MISSES):.1f}%")
except Exception:
    pass

def cmd_smoke_test(args):
    # 1) Small resolve
    s_dict = {
        "num_players": 6,
        "stage": "preflop",
        "players": [
            {"stack": 10000, "hole": [12, 25]},
            {"stack": 10000}, {"stack": 10000}, {"stack": 10000}, {"stack": 10000}, {"stack": 10000}
        ],
        "board": [], "to_act": 0, "current_bet": 100, "min_raise": 100, "pot": 150, "btn": 0
    }
    s = state_from_json(s_dict)
    strat = depth_limited_resolve(s, value_fn=(lambda st,pl: 0.0), cfr_iters=5, depth_limit=2)
    resolve_out = {str(k): float(v) for k, v in strat.items()}

    # 2) One strict BR sample (empty blueprint)
    bp = {}
    value_fn = (lambda st, pl: 0.0)
    ss = new_hand(btn=0, num_players=6)
    for _ in range(2):
        if is_chance_node(ss):
            ss = chance_step(ss)
    memo = {}
    br = _br_value(ss, 0, bp, value_fn, cfr_iters=10, depth_limit=2, depth_cap=2, memo=memo)
    sv = _sigma_value(ss, 0, bp, value_fn, cfr_iters=10, depth_limit=2, depth_cap=2, memo={})
    br_out = {"br": br, "sigma": sv, "gap": max(0.0, br - sv)}

    # 3) Tiny evaluation: 2 hands vs random baseline
    winnings = []
    btn = 0
    for _ in range(2):
        res, btn = _play_hand({}, (lambda s,p: 0.0), _baseline_random_action, num_players=6, btn=btn)
        winnings.append(res[0])
    eval_out = {"hands": len(winnings), "winnings": winnings}

    print(json.dumps({
        "resolve": resolve_out,
        "br": br_out,
        "eval": eval_out
    }, indent=2))

def cmd_quickcheck(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    # 1) Train exactly 1 iteration and save a tiny blueprint
    tab = {"regret": defaultdict(lambda: defaultdict(float)),
           "strategy_sum": defaultdict(lambda: defaultdict(float))}
    btn = 0
    s = new_hand(btn=btn, num_players=args.num_players)
    mccfr_train_iteration(tab, s, value_fn=None, depth_limit=None, iteration=1, linear_weighting=True)
    # Dump avg strategy
    strategy = {}
    for infoset, acts in tab["strategy_sum"].items():
        denom = sum(acts.values()) + 1e-12
        strategy[infoset] = {str(a): float(v/denom) for a, v in acts.items()}
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(strategy, f)
    # 2) Evaluate a few hands vs random
    winnings = []
    bp = strategy
    value_fn = (lambda st, pl: 0.0)
    btn = 0
    for _ in range(args.hands):
        res, btn = _play_hand(bp, value_fn, _baseline_random_action, num_players=args.num_players, btn=btn)
        winnings.append(res[0])
    
    report_results(winnings, BIG_BLIND, "quickcheck")
    print(json.dumps({"saved": args.save}, indent=2))

def cmd_engine_sanity(args):
    results = []
    def record(name:str, ok:bool, info:str=""):
        results.append({"test": name, "ok": bool(ok), "info": info})

    # Test 1: blinds, deal, to_act
    try:
        btn = 0; n=6
        s = new_hand(btn=btn, num_players=n)
        sb_idx = (btn + 1) % n; bb_idx = (btn + 2) % n
        ok_cards = all(len(p["hole"]) == 2 for p in s["players"])
        ok_blinds = (s["players"][sb_idx]["invested"] == s["sb"]) and (s["players"][bb_idx]["invested"] == s["bb"]) and (s["current_bet"] == s["bb"]) and (s["min_raise"] == s["bb"]) and (s["to_act"] == (bb_idx + 1) % n)
        record("blinds_and_deal", ok_cards and ok_blinds)
    except Exception as e:
        record("blinds_and_deal", False, str(e))

    # Test 2: check not legal facing a bet
    try:
        s = new_hand(btn=0, num_players=6)
        la = legal_actions(s)
        record("no_check_facing_bet", ('X',) not in la)
    except Exception as e:
        record("no_check_facing_bet", False, str(e))

    # Test 3: min-raise clamp and update
    try:
        s = new_hand(btn=0, num_players=6)
        prev_cb = s["current_bet"]
        # attempt an under-min raise to force clamp
        s2 = step(s, ('R', prev_cb + s["bb"] - 1))
        ok_target_ge_min = (s2["current_bet"] >= prev_cb + s["bb"])  # min raise from bb to 2*bb
        ok_min_raise_updated = (s2["min_raise"] >= s["bb"])  # at least bb
        record("min_raise_rule", ok_target_ge_min and ok_min_raise_updated)
    except Exception as e:
        record("min_raise_rule", False, str(e))

    # Test 4: all-in removes actions for that player
    try:
        s = new_hand(btn=0, num_players=6)
        me = s["to_act"]
        s["players"][me]["stack"] = 0
        s["players"][me]["all_in"] = True
        la = legal_actions(s)
        record("allin_no_actions", len(la) == 0)
    except Exception as e:
        record("allin_no_actions", False, str(e))

    # Test 5: side pot distribution sanity
    try:
        s = make_state(num_players=3, btn=0)
        # Set river and full board
        s["stage_idx"] = 3
        s["board"] = [0, 1, 2, 3, 4]
        # Players with different investments
        for i in range(3):
            s["players"][i]["hole"] = [5 + i, 10 + i]
            s["players"][i]["folded"] = False
        # Emulate investments: P0 all-in 300, P1 500, P2 1000
        invs = [300, 500, 1000]
        total_before = sum(p["stack"] for p in s["players"]) + sum(invs)
        for i,amt in enumerate(invs):
            s["players"][i]["total_invested"] = amt
            s["players"][i]["invested"] = 0
        _showdown(s)
        total_after = sum(p["stack"] for p in s["players"]) + s["pot"] + sum(p["invested"] for p in s["players"]) + sum(p["total_invested"] for p in s["players"])
        record("sidepot_conservation", abs(total_before - total_after) < 1e-6)
    except Exception as e:
        record("sidepot_conservation", False, str(e))

    passed = sum(1 for r in results if r["ok"]) ; failed = len(results) - passed
    print(json.dumps({"passed": passed, "failed": failed, "results": results}, indent=2))

# -------- Strict Best-Response Exploitability --------
def _sigma_from_blueprint_or_resolve(s, blueprint: Dict, value_fn, cfr_iters: int, depth_limit: int) -> Dict[Tuple, float]:
    legal = legal_actions(s)
    if not legal:
        return {}
    # Try online resolve first (with blueprint warm-start if available)
    try:
        strat = depth_limited_resolve(s, value_fn=value_fn, cfr_iters=cfr_iters, depth_limit=depth_limit, blueprint=blueprint)
        if strat:
            # Filter to legal and renormalize
            acts = {a: p for a, p in strat.items() if a in legal and p > 0}
            z = sum(acts.values())
            if z > 0:
                return {a: v / z for a, v in acts.items()}
    except Exception:
        pass
    # Fallback to blueprint
    infoset = info_key(s, s["to_act"]) if not USE_ABSTRACTION else abstraction_key(s, s["to_act"])  # consistent key
    if infoset in blueprint:
        options = []
        probs = []
        for k, v in blueprint[infoset].items():
            ks = k.replace('[', '(').replace(']', ')').replace('"', '').replace("'", '')
            if ks.startswith("(R,"):
                num = ''.join(ch for ch in ks if ch.isdigit())
                a = ('R', int(num))
            elif ks.startswith("(C"):
                a = ('C',)
            elif ks.startswith("(X"):
                a = ('X',)
            elif ks.startswith("(F"):
                a = ('F',)
            else:
                continue
            if a in legal:
                options.append(a); probs.append(float(v))
        if options:
            z = sum(probs)
            if z > 0:
                return {a: p / z for a, p in zip(options, probs)}
    # Uniform over legal as last resort
    return {a: 1.0 / len(legal) for a in legal}

def _br_value(s, target: int, blueprint: Dict, value_fn, cfr_iters: int, depth_limit: int, depth_cap: int, memo: Dict) -> float:
    key = (public_state_key(s), tuple(tuple(p["hole"]) for p in s["players"]), s["to_act"], target, depth_cap)
    if key in memo:
        return memo[key]
    if is_chance_node(s):
        v = _br_value(chance_step(s), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap, memo)
        memo[key] = v
        return v
    if is_terminal(s):
        v = terminal_value(s, target)
        memo[key] = v
        return v
    if depth_cap <= 0:
        v = rollout_value(s, target, value_fn=value_fn)
        memo[key] = v
        return v
    player = s["to_act"]
    legal = legal_actions(s)
    if not legal:
        v = terminal_value(s, target)
        memo[key] = v
        return v
    if player == target:
        best = -1e18
        for a in legal:
            v = _br_value(step(s, a), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap - 1, memo)
            if v > best:
                best = v
        memo[key] = best
        return best
    else:
        sigma = _sigma_from_blueprint_or_resolve(s, blueprint, value_fn, cfr_iters, depth_limit)
        exp = 0.0
        for a, p in sigma.items():
            if p <= 0:
                continue
            exp += p * _br_value(step(s, a), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap - 1, memo)
        memo[key] = exp
        return exp

def _sigma_value(s, target: int, blueprint: Dict, value_fn, cfr_iters: int, depth_limit: int, depth_cap: int, memo: Dict) -> float:
    key = (public_state_key(s), tuple(tuple(p["hole"]) for p in s["players"]), s["to_act"], target, depth_cap, 'sigma')
    if key in memo:
        return memo[key]
    if is_chance_node(s):
        v = _sigma_value(chance_step(s), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap, memo)
        memo[key] = v
        return v
    if is_terminal(s):
        v = terminal_value(s, target)
        memo[key] = v
        return v
    if depth_cap <= 0:
        v = rollout_value(s, target, value_fn=value_fn)
        memo[key] = v
        return v
    sigma = _sigma_from_blueprint_or_resolve(s, blueprint, value_fn, cfr_iters, depth_limit)
    exp = 0.0
    for a, p in sigma.items():
        if p <= 0:
            continue
        exp += p * _sigma_value(step(s, a), target, blueprint, value_fn, cfr_iters, depth_limit, depth_cap - 1, memo)
    memo[key] = exp
    return exp

def _sample_state(num_players: int) -> dict:
    s = new_hand(btn=random.randrange(num_players), num_players=num_players)
    # random prefix to diversify
    for _ in range(random.randint(0, 6)):
        if is_chance_node(s):
            s = chance_step(s)
        else:
            la = legal_actions(s)
            if not la:
                break
            s = step(s, random.choice(la))
    return s

def cmd_exploitability_strict(args):
    bp = load_blueprint(args.blueprint)
    # value function (optional)
    if args.value and os.path.exists(args.value):
        value_fn = make_value_fn_from_dp(args.value) if args.value.endswith('.pt') else make_value_fn(load_value_params(args.value))
    else:
        value_fn = (lambda s, p: 0.0)
    samples = int(args.samples)
    num_players = int(args.num_players)
    depth_cap = int(args.depth_cap)
    cfr_iters = max(RESOLVE_ITERS, int(args.resolve_iters))
    depth_limit = max(RESOLVE_DEPTH, int(args.resolve_depth))
    # Assert non-empty sample size
    if samples <= 0:
        raise AssertionError("Exploitability: пустая выборка (samples <= 0)")
    total = 0.0
    used = 0
    br_calls = 0
    sigma_calls = 0
    for _ in trange(samples, desc="BR exploitability"):
        s = _sample_state(num_players)
        player = random.randrange(num_players)
        memo = {}
        try:
            br = _br_value(s, player, bp, value_fn, cfr_iters, depth_limit, depth_cap, memo); br_calls += 1
            sv = _sigma_value(s, player, bp, value_fn, cfr_iters, depth_limit, depth_cap, memo); sigma_calls += 1
            total += max(0.0, br - sv)
            used += 1
        except Exception:
            continue
    if used == 0:
        print(json.dumps({"error": "no_valid_samples", "samples_requested": samples, "depth_cap": depth_cap}))
        return
    avg = total / used
    mbb = (avg / BIG_BLIND) * 1000.0
    print(json.dumps({
        "samples_requested": samples,
        "samples_used": used,
        "avg_exploitability": avg,
        "mbb_per_hand": mbb,
        "resolve_iters": cfr_iters,
        "resolve_depth": depth_limit,
        "depth_cap": depth_cap,
        "br_calls": br_calls,
        "sigma_calls": sigma_calls
    }, indent=2))

def cmd_auto_pluribus(args):
    # One-command end-to-end pipeline with strong defaults
    from types import SimpleNamespace as NS
    base = os.path.join(PROJECT_ROOT)
    os.makedirs(f"{base}/data", exist_ok=True)
    os.makedirs(f"{base}/ckpts", exist_ok=True)
    # 1) Self-play dataset
    sp_args = NS(
        episodes=int(args.episodes),
        out=f"{base}/data/selfplay_big.npz",
        seed=123,
        num_players=6,
        bet_sizes_preflop=None,
        bet_sizes_postflop=None
    )
    cmd_selfplay_generate(sp_args)
    # 2) Abstraction (potential-aware)
    global PA_ABS
    PA_ABS = True
    build_abstraction_from_dataset(
        data_path=sp_args.out,
        hole_k=int(args.hole_k),
        board_k=int(args.board_k),
        iters=int(args.abs_iters),
        save=f"{base}/abs_big.npz",
    )
    # 3) Value net (distributed)
    train_value_distributed(
        data_path=sp_args.out,
        save_path=f"{base}/value_dp.pt",
        epochs=int(args.value_epochs),
        batch=int(args.value_batch),
        lr=float(args.value_lr),
    )
    # 4) Distributed MCCFR with periodic monitoring and optional H2H vs previous blueprint
    tb_args = NS(
        iterations=int(args.iterations),
        num_players=6,
        num_workers=int(args.num_workers),
        save=f"{base}/blueprint_dist.json",
        save_sum=f"{base}/blueprint_sums.json",
        bet_sizes_preflop=None,
        bet_sizes_postflop=None,
        abstraction=f"{base}/abs_big.npz",
        use_abstraction=True,
        dynamic_abstraction=True,
        chunk=max(1, int(args.iterations)//max(1,int(args.num_workers))),
        checkpoint_dir=f"{base}/ckpts",
        eval_every_chunk=True,
        eval_hands=int(args.eval_hands),
        eval_num_players=6,
        eval_baseline="random",
        eval_value=f"{base}/value_dp.pt",
        eval_abstraction=f"{base}/abs_big.npz",
        h2h_every_chunk=bool(args.prev_blueprint is not None),
        h2h_blueprint=args.prev_blueprint,
        h2h_hands=int(args.h2h_hands),
        h2h_resolve_iters=int(args.h2h_resolve_iters),
        h2h_resolve_depth=int(args.h2h_resolve_depth),
        resolve_determs=int(args.resolve_determs),
        resolve_warmstart=int(args.resolve_warmstart),
        hole_eval_samples=int(args.hole_eval_samples),
        kl_alpha=float(args.kl_alpha),
        resolve_parallel=True,
        posterior_from_blueprint=True,
    )
    cmd_train_blueprint_dist(tb_args)
    # 5) Final evaluations
    ev_common = dict(
        blueprint=tb_args.save,
        value=f"{base}/value_dp.pt",
        abstraction=f"{base}/abs_big.npz",
        hands=int(args.final_hands),
        num_players=6,
        resolve_iters=int(args.final_resolve_iters),
        resolve_depth=int(args.final_resolve_depth),
        use_abstraction=True,
        dynamic_abstraction=True,
        resolve_determs=int(args.resolve_determs),
        resolve_warmstart=int(args.resolve_warmstart),
        hole_eval_samples=int(args.hole_eval_samples),
        kl_alpha=float(args.kl_alpha),
    )
    # vs random
    cmd_evaluate(NS(baseline="random", bet_sizes_preflop=None, bet_sizes_postflop=None, **ev_common))
    # vs tight
    cmd_evaluate(NS(baseline="tight", bet_sizes_preflop=None, bet_sizes_postflop=None, **ev_common))
    # exploitability
    cmd_exploitability_strict(NS(blueprint=tb_args.save, value=f"{base}/value_dp.pt", samples=int(args.expl_samples), num_players=6, depth_cap=int(args.expl_depth_cap), resolve_iters=int(args.final_resolve_iters), resolve_depth=int(args.final_resolve_depth)))

def parse_action_str(a_str):
    a_str = a_str.strip().lower()
    if a_str in ['f','fold']: return ('F',)
    if a_str in ['x','check']: return ('X',)
    if a_str in ['c','call']: return ('C',)
    if a_str.startswith('r'):
        num=''.join(ch for ch in a_str if ch.isdigit())
        if num=='': raise ValueError("Use 'r <amount>' e.g., r 600")
        return ('R', int(num))
    raise ValueError("Unknown action. Use: f/x/c or r <amount>")

def cmd_play_cli(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    blueprint = load_blueprint(args.blueprint)
    value_fn = None
    if args.value:
        params = load_value_params(args.value)
        value_fn = make_value_fn(params)
    else:
        value_fn = lambda s,p: 0.0
    btn=0
    s = new_hand(btn=btn, num_players=args.num_players)
    print(f"[bold magenta]Hydra6 single-file CLI[/bold magenta]  (you are seat {args.human_seat})")
    print(f"Bot sits at seat {args.bot_seat}.")
    human=args.human_seat; bot=args.bot_seat
    while True:
        while is_chance_node(s): s = chance_step(s)
        if is_terminal(s):
            print("[bold]Hand finished[/bold]. Stacks:")
            for i,p in enumerate(s["players"]): print(f"Seat {i}: stack={p['stack']}")
            btn = (btn + 1) % s["num_players"]
            s = new_hand(btn=btn, num_players=args.num_players); continue
        cp = s["to_act"]
        if cp == human:
            print(f"[cyan]Stage:[/cyan] {s['stage_idx']}  [cyan]Board:[/cyan] {s['board']}")
            print(f"[cyan]Pot:[/cyan] {pot_size(s)}  [cyan]Current bet:[/cyan] {s['current_bet']}")
            me = s["players"][human]
            print(f"[green]Your hole:[/green] {me['hole']}  [green]Stack:[/green] {me['stack']}  [green]Invested:[/green] {me['invested']}")
            la = legal_actions(s); print(f"Legal: {la}")
            try:
                a = parse_action_str(Prompt.ask("Your action (f/x/c or r <amt>)"))
            except Exception as e:
                print(f"[red]{e}[/red]"); continue
            s = step(s, a)
        elif cp == bot:
            a = choose_bot_action(s, blueprint, value_fn, resolve_iters=args.resolve_iters, resolve_depth=args.resolve_depth)
            print(f"[yellow]Bot action:[/yellow] {a}")
            s = step(s, a)
        else:
            la = legal_actions(s); a=None
            if la:
                raises=[x for x in la if x[0]=='R']
                if raises and random.random()<0.1: a=random.choice(raises)
                else: a=('C',) if ('C',) in la else ('X',) if ('X',) in la else random.choice(la)
            if a is not None: s = step(s,a)

def state_from_json(d:dict):
    s = make_state(num_players=d["num_players"], sb=d.get("sb",50), bb=d.get("bb",100), ante=d.get("ante",0), btn=d.get("btn",0))
    s["players"] = [{
        "stack": int(p.get("stack", STARTING_STACK)),
        "invested": int(p.get("invested", 0)),
        "total_invested": int(p.get("total_invested", 0)),
        "folded": bool(p.get("folded", False)),
        "all_in": bool(p.get("all_in", False)),
        "hole": list(p.get("hole", []))
    } for p in d["players"]]
    s["stage_idx"] = {"preflop":0,"flop":1,"turn":2,"river":3}[d["stage"]]
    s["board"] = list(d.get("board", []))
    s["to_act"] = int(d["to_act"])
    s["current_bet"] = int(d.get("current_bet", 0))
    s["min_raise"] = int(d.get("min_raise", s["bb"]))
    s["pot"] = int(d.get("pot", 0))
    return s

def cmd_resolve_from_json(args):
    set_bet_sizes(args.bet_sizes_preflop, args.bet_sizes_postflop)
    with open(args.situation,"r") as f: d=json.load(f)
    s = state_from_json(d)
    value_fn=None
    if args.value:
        params = load_value_params(args.value)
        value_fn = make_value_fn(params)
    else:
        value_fn = lambda st,pl: 0.0
    if VALUE_FN_ACTIVE:
        try:
            print(json.dumps({"value_fn": {"active": True, "source": VALUE_FN_SOURCE}}))
        except Exception:
            pass
    # Apply resolve enhancements if provided
    set_resolve_enhancements(getattr(args, 'resolve_determs', None), getattr(args, 'resolve_warmstart', None), getattr(args, 'hole_eval_samples', None))
    set_resolve_regularization(getattr(args, 'kl_alpha', None))
    set_resolve_modes(getattr(args, 'resolve_parallel', None), getattr(args, 'posterior_from_blueprint', None))
    bp_json = load_blueprint(args.blueprint) if args.blueprint else None
    strat = depth_limited_resolve(s, value_fn=value_fn, cfr_iters=max(args.cfr_iters, RESOLVE_ITERS), depth_limit=max(args.depth, RESOLVE_DEPTH), blueprint=bp_json)
    if args.blueprint and not strat:
        bp = load_blueprint(args.blueprint)
        infoset = f"{public_state_key(s)}|priv:{private_obs(s, s['to_act'])}|p:{s['to_act']}"
        strat = bp.get(infoset, {})
    out = {str(k): float(v) for k,v in strat.items()}
    print(json.dumps(out, indent=2))

# -------- Smoke Tests for Actions --------
def _compute_min_raise(current_bet, invested, bb):
    return max(bb, invested - current_bet)

def _build_raise_target(pot, current_bet, invested, bb, frac):
    minr = _compute_min_raise(current_bet, invested, bb)
    pot_target = int(round(pot * frac))
    target = max(current_bet + minr, pot_target)
    return max(target, current_bet + minr), minr

def cmd_smoke_actions(args=None):
    failures = []
    bb = 100
    # (1) min-raise invariant across random states
    for _ in range(200):
        pot = random.randint(150, 10000)
        current_bet = random.randint(0, pot // 2)
        invested = current_bet + random.randint(0, pot // 3)
        frac = random.choice(BET_FRACTIONS_POSTFLOP)
        tgt, minr = _build_raise_target(pot, current_bet, invested, bb, frac)
        if minr < bb:
            failures.append(f"min-raise<{bb}: minr={minr}, state(pot={pot}, cb={current_bet}, inv={invested})")
        if tgt < current_bet + minr:
            failures.append(f"target<cb+minr: tgt={tgt}, cb={current_bet}, minr={minr}")
    # (2) POT/JAM nodes expected by config
    if ACTION_ENSURE_POT_NODE and (0.99 not in BET_FRACTIONS_POSTFLOP and 1.0 not in BET_FRACTIONS_POSTFLOP):
        failures.append("POT node expected but no ~1.0 fraction in POSTFLOP ladder")
    if ACTION_ENSURE_JAM_NODE and not ALLOW_ALLIN:
        failures.append("JAM node expected but ALLOW_ALLIN=False")
    # (3) All-in freeze assumptions: require raises enabled
    if ACTION_ENSURE_JAM_NODE and RAISE_CAP == 0:
        failures.append("All-in freeze inconclusive: RAISE_CAP==0 disables raises")

    if failures:
        print("[SMOKE] FAIL")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("[SMOKE] OK: min-raise invariant, POT/JAM presence, all-in freeze assumptions passed.")
    return 0

# -------- CLI --------
def main():
    p = argparse.ArgumentParser(description="Hydra6 single-file")
    sp = p.add_subparsers(dest="cmd", required=True)

    # selfplay
    a = sp.add_parser("selfplay_generate")
    a.add_argument("--episodes", type=int, default=2000)
    a.add_argument("--out", type=str, default="data/selfplay.npz")
    a.add_argument("--seed", type=int, default=123)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.set_defaults(func=cmd_selfplay_generate)

    # generate_cfv
    a = sp.add_parser("generate_cfv")
    a.add_argument("--samples", type=int, default=50000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--blueprint", type=str, default=None)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--resolve-iters", type=int, default=1000)
    a.add_argument("--resolve-depth", type=int, default=5)
    a.add_argument("--depth-cap", type=int, default=5)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--resolve-determs", type=int, default=None)
    a.add_argument("--resolve-warmstart", type=int, default=None)
    a.add_argument("--hole-eval-samples", type=int, default=None)
    a.add_argument("--kl-alpha", type=float, default=None)
    a.add_argument("--out", type=str, default="data/cfv.npz")
    a.set_defaults(func=cmd_generate_cfv)

    # train_value
    a = sp.add_parser("train_value")
    a.add_argument("--data", type=str, required=True)
    a.add_argument("--save", type=str, default="value_single.pt")
    a.add_argument("--epochs", type=int, default=5)
    a.add_argument("--batch", type=int, default=512)
    a.add_argument("--lr", type=float, default=3e-4)
    a.set_defaults(func=cmd_train_value)

    # train_value_dist (multi-GPU)
    def _cmd_train_value_dist(args):
        return train_value_distributed(args.data, save_path=args.save, epochs=args.epochs, batch=args.batch, lr=args.lr)
    a = sp.add_parser("train_value_dist")
    a.add_argument("--data", type=str, required=True)
    a.add_argument("--save", type=str, default="value_dp.pt")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--batch", type=int, default=8192)
    a.add_argument("--lr", type=float, default=1e-3)
    a.set_defaults(func=_cmd_train_value_dist)

    # train_blueprint
    a = sp.add_parser("train_blueprint")
    a.add_argument("--iterations", type=int, default=20000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--save", type=str, default="blueprint.json")
    a.add_argument("--save-sum", type=str, default=None)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--use-abstraction", action="store_true",
                   help="Force-enable abstraction (otherwise autoloads if file exists).")
    a.add_argument("--no-abstraction", action="store_true",
                   help="Force-disable abstraction (overrides autoload).")
    a.add_argument("--regret-decay", type=float, default=0.0)
    a.add_argument("--slice", type=int, default=10000, help="LCFR/DCFR discount slice (iters)")
    a.set_defaults(func=cmd_train_blueprint)

    # train_blueprint_dist (multi-process)
    a = sp.add_parser("train_blueprint_dist")
    a.add_argument("--iterations", type=int, default=100000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 1))
    a.add_argument("--save", type=str, default="blueprint_dist.json")
    a.add_argument("--save-sum", type=str, default=None)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--abstraction", type=str, default=None)
    a.add_argument("--use-abstraction", action="store_true")
    a.add_argument("--dynamic-abstraction", action="store_true")
    a.add_argument("--chunk", type=int, default=None)
    a.add_argument("--checkpoint-dir", type=str, default=None)
    a.add_argument("--ckpt-keep-every", type=int, default=None)
    a.add_argument("--ckpt-max-gb", type=float, default=None)
    a.add_argument("--milestones-jsonl", type=str, default=None)
    a.add_argument("--logs-dir", type=str, default=None)
    a.add_argument("--logs-keep-days", type=int, default=None)
    a.add_argument("--logs-max-gb", type=float, default=None)
    # Auto-monitoring options
    a.add_argument("--eval-every-chunk", action="store_true")
    a.add_argument("--eval-hands", type=int, default=1000)
    a.add_argument("--eval-num-players", type=int, default=6)
    a.add_argument("--eval-baseline", type=str, choices=["random","tight"], default="random")
    a.add_argument("--eval-value", type=str, default=None)
    a.add_argument("--eval-abstraction", type=str, default=None)
    # Periodic head-to-head options
    a.add_argument("--h2h-every-chunk", action="store_true")
    a.add_argument("--h2h-blueprint", type=str, default=None)
    a.add_argument("--h2h-hands", type=int, default=1000)
    a.add_argument("--h2h-resolve-iters", type=int, default=None)
    a.add_argument("--h2h-resolve-depth", type=int, default=None)
    a.add_argument("--resolve-determs", type=int, default=None)
    a.add_argument("--resolve-warmstart", type=int, default=None)
    a.add_argument("--hole-eval-samples", type=int, default=None)
    a.add_argument("--kl-alpha", type=float, default=None)
    a.add_argument("--aivat-lite", action="store_true",
                   help="Enable AB-MIVAT style variance reduction in match reports.")
    a.add_argument("--resolve-parallel", action="store_true")
    a.add_argument("--posterior-from-blueprint", action="store_true")
    a.add_argument("--milestone-interval", type=int, default=None)
    a.add_argument("--milestone-dir", type=str, default=None)
    # Pruning and CFR controls
    a.add_argument("--neg-prune-prob", type=float, default=None, help="95/5 negative-regret prune prob")
    a.add_argument("--neg-prune-threshold", type=float, default=None, help="Regret threshold for pruning")
    a.add_argument("--regret-floor", type=float, default=None, help="Floor for regrets")
    a.add_argument("--slice", type=int, default=10000, help="LCFR/DCFR discount slice (iters)")
    a.set_defaults(func=cmd_train_blueprint_dist)

    # merge
    a = sp.add_parser("merge_strategy_sums")
    a.add_argument("--inputs", nargs="+", required=True)
    a.add_argument("--out-blueprint", type=str, default="blueprint_merged.json")
    a.add_argument("--out-sum", type=str, default=None)
    a.set_defaults(func=cmd_merge_sums)

    # play_cli
    a = sp.add_parser("play_cli")
    a.add_argument("--blueprint", type=str, required=True)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--bot-seat", type=int, default=2)
    a.add_argument("--human-seat", type=int, default=0)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--resolve-iters", type=int, default=None)
    a.add_argument("--resolve-depth", type=int, default=None)
    a.set_defaults(func=cmd_play_cli)

    # evaluate
    a = sp.add_parser("evaluate")
    a.add_argument("--blueprint", type=str, required=True)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--abstraction", type=str, default=None)
    a.add_argument("--baseline", type=str, choices=["random","tight"], default="random")
    a.add_argument("--hands", type=int, default=5000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--resolve-iters", type=int, default=None)
    a.add_argument("--resolve-depth", type=int, default=None)
    a.add_argument("--resolve-determs", type=int, default=None)
    a.add_argument("--resolve-warmstart", type=int, default=None)
    a.add_argument("--hole-eval-samples", type=int, default=None)
    a.add_argument("--kl-alpha", type=float, default=None)
    a.add_argument("--aivat-lite", action="store_true",
                   help="Enable AB-MIVAT style variance reduction in match reports.")
    a.add_argument("--resolve-parallel", action="store_true")
    a.add_argument("--posterior-from-blueprint", action="store_true")
    a.add_argument("--opp-blueprint", type=str, default=None)
    a.add_argument("--villain-milestones", nargs='*', default=None)
    a.add_argument("--h2h-hands", type=int, default=None)
    a.add_argument("--h2h-resolve-iters", type=int, default=None)
    a.add_argument("--h2h-resolve-depth", type=int, default=None)
    a.add_argument("--hero", type=int, default=0)
    a.add_argument("--use-abstraction", action="store_true",
                   help="Force-enable abstraction (otherwise autoloads if file exists).")
    a.add_argument("--no-abstraction", action="store_true",
                   help="Force-disable abstraction (overrides autoload).")
    a.set_defaults(func=cmd_evaluate)

    # strict exploitability (best response)
    a = sp.add_parser("exploitability_strict")
    a.add_argument("--blueprint", type=str, required=True)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--samples", type=int, default=100000)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--depth-cap", type=int, default=6)
    a.add_argument("--resolve-iters", type=int, default=1000)
    a.add_argument("--resolve-depth", type=int, default=6)
    a.set_defaults(func=cmd_exploitability_strict)

    # smoke_test
    a = sp.add_parser("smoke_test")
    a.set_defaults(func=cmd_smoke_test)

    # smoke_actions
    a = sp.add_parser("smoke_actions")
    a.set_defaults(func=cmd_smoke_actions)

    # quickcheck: 1-iteration train + tiny eval
    a = sp.add_parser("quickcheck")
    a.add_argument("--save", type=str, default=os.path.join(PROJECT_ROOT, "blueprint_quick.json"))
    a.add_argument("--hands", type=int, default=4)
    a.add_argument("--num-players", type=int, default=6)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.set_defaults(func=cmd_quickcheck)

    # engine_sanity: validate core poker rules
    a = sp.add_parser("engine_sanity")
    a.set_defaults(func=cmd_engine_sanity)

    # resolve_from_json
    a = sp.add_parser("resolve_from_json")
    a.add_argument("--situation", type=str, required=True)
    a.add_argument("--blueprint", type=str, default=None)
    a.add_argument("--value", type=str, default=None)
    a.add_argument("--cfr-iters", type=int, default=200)
    a.add_argument("--depth", type=int, default=3)
    a.add_argument("--bet-sizes-preflop", type=str, default=None)
    a.add_argument("--bet-sizes-postflop", type=str, default=None)
    a.add_argument("--resolve-determs", type=int, default=None)
    a.add_argument("--resolve-warmstart", type=int, default=None)
    a.add_argument("--hole-eval-samples", type=int, default=None)
    a.add_argument("--kl-alpha", type=float, default=None)
    a.add_argument("--aivat-lite", action="store_true",
                   help="Enable AB-MIVAT style variance reduction in match reports.")
    a.add_argument("--resolve-parallel", action="store_true")
    a.add_argument("--posterior-from-blueprint", action="store_true")
    a.set_defaults(func=cmd_resolve_from_json)

    # abstraction build/load
    a = sp.add_parser("build_abstraction")
    a.add_argument("--data", type=str, required=True)
    a.add_argument("--hole-k", type=int, default=256)
    a.add_argument("--board-k", type=int, default=128)
    a.add_argument("--iters", type=int, default=25)
    a.add_argument("--save", type=str, default="abs.npz")
    a.add_argument("--potential-aware", action="store_true")
    def _cmd_build_abs(args):
        global PA_ABS
        PA_ABS = bool(args.potential_aware)
        return build_abstraction_from_dataset(args.data, args.hole_k, args.board_k, args.iters, args.save)
    a.set_defaults(func=_cmd_build_abs)

    a = sp.add_parser("load_abstraction")
    a.add_argument("--path", type=str, required=True)
    a.set_defaults(func=lambda args: load_abstraction(args.path))

    # auto_pluribus: one-command end-to-end pipeline
    a = sp.add_parser("auto_pluribus")
    a.add_argument("--episodes", type=int, default=3000000)
    a.add_argument("--hole-k", type=int, default=1024)
    a.add_argument("--board-k", type=int, default=512)
    a.add_argument("--abs-iters", type=int, default=30)
    a.add_argument("--value-epochs", type=int, default=30)
    a.add_argument("--value-batch", type=int, default=16384)
    a.add_argument("--value-lr", type=float, default=3e-4)
    a.add_argument("--iterations", type=int, default=2000000)
    a.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() or 1))
    a.add_argument("--eval-hands", type=int, default=2000)
    a.add_argument("--h2h-hands", type=int, default=1000)
    a.add_argument("--h2h-resolve-iters", type=int, default=600)
    a.add_argument("--h2h-resolve-depth", type=int, default=5)
    a.add_argument("--resolve-determs", type=int, default=8)
    a.add_argument("--resolve-warmstart", type=int, default=50)
    a.add_argument("--hole-eval-samples", type=int, default=2000)
    a.add_argument("--kl-alpha", type=float, default=0.2)
    a.add_argument("--final-hands", type=int, default=20000)
    a.add_argument("--final-resolve-iters", type=int, default=800)
    a.add_argument("--final-resolve-depth", type=int, default=5)
    a.add_argument("--expl-samples", type=int, default=50000)
    a.add_argument("--expl-depth-cap", type=int, default=6)
    a.add_argument("--prev-blueprint", type=str, default=None)
    a.set_defaults(func=cmd_auto_pluribus)

    args = p.parse_args()
    random.seed(42)
    if hasattr(args, "seed"): random.seed(args.seed)
    return args.func(args) if hasattr(args, 'func') else None

if __name__ == "__main__":
    main()
