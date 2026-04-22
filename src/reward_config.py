"""
reward_config.py
~~~~~~~~~~~~~~~~
Centralised reward hyperparameters for the Pokémon RL training system.

All weights used by PokemonEnv._compute_reward() are defined here as
named constants.  Import this module anywhere you need to inspect, log,
or override reward shaping without editing pokemon_env.py.

USAGE
-----
    from src.reward_config import DEFAULT_WEIGHTS, RewardExplainer

    # Inspect weights
    print(DEFAULT_WEIGHTS)

    # Pretty-print a reward breakdown from env.step()
    obs, reward, done, trunc, info = env.step(action)
    RewardExplainer.print(info["reward_breakdown"])

REWARD ARCHITECTURE — 15 components, 3 tiers
─────────────────────────────────────────────
Tier 1  TERMINAL (dominant — drives the objective)
  • terminal_win / terminal_loss  ±1.00   Win or lose the battle
  • ko_bonus / faint_penalty      ±0.22   Individual KO events
  • anti_burst_penalty            −0.15   Discounts fast wins (≤4 turns)

Tier 2  PER-TURN STRATEGIC (accumulate over episode length)
  • damage_reward                 ±0.12   HP differential each turn
  • survival_bonus                +0.008  Per-turn alive reward
  • hp_lead_bonus                 +0.015  Proportional HP advantage
  • consecutive_bonus             +0.030  Sustained dominance (compound)
  • stall_penalty                 −0.015  True stall turns only

Tier 3  POSITIONAL / SHAPING (gradient toward competitive play)
  • matchup_shaping               ±0.035  Type-advantage gradient
  • bad_stay_penalty              −0.045  Losing matchup + threat
  • temporal_risk                 −0.040  KO-threat deterrent
  • momentum_reward               ±0.020  EMA damage-differential
  • move_quality                  ±0.060  SE/immunity + accuracy
  • smart_switch                  ±0.060  Switch quality delta

TUNING PHILOSOPHY
─────────────────
• Tier 1 weights are the LEARNING SIGNAL.  Do not reduce below ~0.5×.
• Tier 2 weights SHAPE EPISODE LENGTH.  Increase to favour longer strategic
  battles; decrease to allow burst play.
• Tier 3 weights guide MOVE SELECTION and POSITIONING.  Safe to tune freely
  — they are small enough that they cannot override Tier 1 objectives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
#  REWARD WEIGHTS DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    """
    All scalar hyperparameters used by PokemonEnv._compute_reward().

    Each field has an inline comment describing its purpose and safe tuning
    range.  Change a field and pass the instance to PokemonEnv (not yet wired
    via kwarg — use as documentation + logging reference for now).

    Default values reflect the production-calibrated settings that produce
    strategic multi-turn battles.  See the class-level docstring for the
    Tier 1 / Tier 2 / Tier 3 architecture.
    """

    # ── Tier 1: Terminal signals ──────────────────────────────────────────────

    terminal_win: float = 1.00
    """Win the battle.  The dominant learning signal — do not reduce below 0.8.
    All other components are shaping terms relative to this anchor."""

    terminal_loss: float = -1.00
    """Lose the battle.  Keep symmetric with terminal_win for zero-sum play."""

    ko_bonus: float = 0.22
    """Reward for knocking out the opponent's active Pokémon.

    Tuning range: [0.10, 0.35].
    Values above 0.35 cause burst-KO exploitation — agent learns to rush
    one-hit KOs rather than building positional advantage.
    Values below 0.10 slow credit assignment for multi-Pokémon battles.
    """

    faint_penalty: float = 0.22
    """Penalty for having the IA's active Pokémon fainted.

    Keep equal to ko_bonus for symmetric zero-sum incentives.
    Reduce independently to make the agent more risk-tolerant.
    """

    anti_burst_penalty: float = 0.20
    """Discount applied to terminal_win when the battle ends in ≤ anti_burst_turns.

    Net fast-win reward: terminal_win − anti_burst_penalty = +0.80 (still clearly positive).
    Moderate penalty: discourages 1-2 hit burst without confusing the win signal.
    Does NOT fire on fast losses (would perversely reward dying fast).
    """

    anti_burst_turns: int = 5
    """Turn threshold for anti_burst_penalty.  Wins in ≤ this many turns are discounted."""

    # ── Tier 2: Per-turn strategic bonuses ───────────────────────────────────

    damage_dealt_k: float = 0.12
    """Per-turn weight on normalised HP removed from the opponent.

    Tuning range: [0.05, 0.20].
    Lower → less burst incentive, slower credit assignment.
    Higher → faster learning but encourages damage-spam over positioning.
    Keep equal to damage_taken_k for symmetric treatment.
    """

    damage_taken_k: float = 0.12
    """Per-turn penalty weight on normalised HP taken by the IA.  Mirror of damage_dealt_k."""

    stall_penalty: float = 0.015
    """Penalty per true-stall turn — turns where BOTH sides deal zero damage.

    Only fires on status-move spam, double miss, or recharge turns.
    Range: [0.010, 0.025].  Small enough to not override strategic status use,
    large enough to break infinite stall loops.
    """

    survival_bonus: float = 0.010
    """Per-turn reward for being alive (hp_ia > 0).

    Accumulates over episode length:
      2-turn battle → +0.016   8-turn battle → +0.064
    First of three compounding 'length rewards'.  Range: [0.004, 0.015].
    """

    hp_lead_k: float = 0.015
    """Weight on max(0, hp_ia − hp_rival) per turn.

    Teaches HP conservation and rewards dominant positioning.
    Zero when the IA is behind (no negative version — temporal_risk handles that).
    Range: [0.008, 0.025].
    """

    consec_adv_k: float = 0.003
    """Per-unit reward on the consecutive-advantage counter (0 .. consec_adv_max).

    Counter grows by +1 each turn the IA holds hp_ia > hp_rival + 0.05.
    Counter decays by −1 when the lead is lost.
    Maximum bonus per turn: consec_adv_k × consec_adv_max = +0.030.
    Cumulative over 10 consecutive turns: +0.165 (compound signal).
    """

    consec_adv_max: int = 10
    """Cap on the consecutive-advantage counter.  Controls maximum compound bonus."""

    # ── Tier 3: Positional / shaping signals ─────────────────────────────────

    matchup_k: float = 0.055
    """Weight on the type-matchup score ∈ [−1, +1] (computed every turn).

    matchup_score = (best_eff_me − best_eff_foe) / 4.0
    Provides a continuous gradient toward favourable type matchups.
    Increased 0.035 → 0.055 to stronger emphasise type-advantage play.
    Range: [0.015, 0.060].
    """

    bad_stay_matchup_k: float = 0.035
    """Penalty weight on |matchup_score| when in a losing matchup without switching.

    Combined max bad-stay penalty: bad_stay_matchup_k × 1.0 + bad_stay_threat_k × 1.0
    = −0.075/turn in the worst matchup under full KO threat.
    Increased 0.020 → 0.035 to more aggressively punish staying in bad matchups.
    """

    bad_stay_threat_k: float = 0.040
    """Additional penalty weight on threat_level when in a losing matchup.
    Increased 0.025 → 0.040 — stronger urgency to escape KO threats.
    See bad_stay_matchup_k above."""

    bad_stay_threshold: float = -0.10
    """matchup_score threshold below which the bad-stay penalty activates.

    Range: [−0.30, −0.05].  Less negative = stricter (fires more often).
    More negative = lenient (only fires in severely bad matchups).
    """

    temporal_risk_k: float = 0.040
    """Per-turn penalty proportional to threat_level (0..1, 1 = KO expected next hit).

    Applied independently of matchup: fires even in a favourable matchup if
    the opponent can OHKO the IA.  Creates urgency to switch before being KO'd.
    Range: [0.020, 0.060].
    """

    momentum_k: float = 0.020
    """Weight on the EMA of (damage_dealt − damage_taken) per turn.

    Positive momentum = IA is pressing; negative = IA is being pressured.
    Applied through tanh so the signal is bounded ± momentum_k regardless of scale.
    """

    momentum_alpha: float = 0.35
    """EMA update coefficient for the momentum signal.

    momentum_t = alpha × delta_t + (1 − alpha) × momentum_{t−1}
    Range: [0.20, 0.50].  0.35 ≈ memory of last 3–4 turns.
    Higher alpha → faster response to recent turns.
    Lower alpha → smoother, longer memory.
    """

    move_quality_k: float = 0.04
    """Weight on the log2-effectiveness component of the move quality signal.

    effectiveness → move_quality:
      immune (0×)  → hard-coded move_immune_pen (−0.10)
      0.25×        → −0.08
      0.5×         → −0.04
      1× (neutral) →  0.00
      2× (SE)      → +0.04
      4× (double SE) → +0.08

    Doubled from 0.02 → 0.04 to strengthen the per-move effectiveness signal.
    This teaches the agent to prefer super-effective moves more aggressively.
    """

    move_immune_pen: float = -0.10
    """Hard penalty for using a move the opponent is fully immune to (0× effectiveness).

    Separate from the log2 curve because log2(0) is undefined.
    Increased −0.06 → −0.10 to strongly discourage immune moves.
    Should be ≤ −2 × move_quality_k to be stronger than the worst non-immune case.
    """
    """Hard penalty for using a move the opponent is fully immune to (0× effectiveness).

    Separate from the log2 curve because log2(0) is undefined.
    Should be ≤ −2 × move_quality_k to be stronger than the worst non-immune case.
    """

    acc_penalty_k: float = 0.008
    """Penalty per percentage point of accuracy below 100%.

    A 50% accuracy move adds: −acc_penalty_k × (1 − 50/100) = −0.004.
    Discourages low-accuracy moves even when they have high power.
    """

    smart_switch_k: float = 0.09
    """Weight on the switch-quality delta when the IA voluntarily switches.

    switch_quality_delta = (new_matchup − old_matchup) + 0.5 × (old_threat − new_threat)

    A genuinely good switch (full matchup flip from −1 to +1) yields +0.18 before
    switch_cost.  After switch_cost (−0.01), net = +0.17 → strong positive incentive.
    A bad switch (matchup worsens) can yield up to −0.09 − 0.01 = −0.10.

    Increased 0.06 → 0.09 to make good switches more rewarding.
    Reduced by switch_fatigue_mult after switch_fatigue_n switches to prevent spam.
    Range: [0.03, 0.10].
    """

    switch_cost: float = 0.01
    """Flat per-switch penalty on voluntary IA switches.

    Provides friction against switch-spam.  smart_switch_k must exceed this
    for a genuinely good switch to be net positive.
    """

    switch_fatigue_n: int = 8
    """Switch count after which the fatigue multiplier starts reducing smart_switch rewards.

    After N switches: fatigue_mult = max(0.70, 1 − 0.30 × (count / N))
    At 8+ switches: multiplier floors at 0.70 — smart switch reward reduced by 30%.
    """


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLETON — default weights shipped with the system
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = RewardWeights()


# ─────────────────────────────────────────────────────────────────────────────
#  REWARD EXPLAINER — human-readable per-step reward breakdown
# ─────────────────────────────────────────────────────────────────────────────

class RewardExplainer:
    """
    Pretty-print the reward breakdown dict returned in info["reward_breakdown"].

    Usage
    -----
        obs, reward, done, trunc, info = env.step(action)
        RewardExplainer.print(info["reward_breakdown"])

        # Or get the string for logging
        text = RewardExplainer.explain(info["reward_breakdown"])
    """

    # Ordered display labels for each reward component key
    _LABELS: Dict[str, str] = {
        # Tier 1 — Terminal
        "terminal_bonus":       "Terminal (win/loss)",
        "anti_burst_penalty":   "Anti-burst discount",
        "ko_bonus":             "KO bonus",
        "faint_penalty":        "Faint penalty",
        "ko_reward":            "KO net (bonus − penalty)",
        # Tier 2 — Per-turn strategic
        "damage_dealt_reward":  "Damage dealt",
        "damage_taken_penalty": "Damage taken",
        "damage_reward":        "Damage net",
        "survival_bonus":       "Survival bonus",
        "hp_lead_bonus":        "HP lead bonus",
        "consecutive_bonus":    "Consecutive advantage",
        "stall_penalty":        "Stall penalty",
        # Tier 3 — Positional
        "matchup_shaping":      "Matchup shaping",
        "bad_stay_penalty":     "Bad-stay penalty",
        "temporal_risk":        "Temporal risk (KO threat)",
        "momentum_reward":      "Momentum (EMA)",
        "move_quality":         "Move quality",
        # Switch
        "smart_switch":         "Smart switch quality",
        "switch_penalty":       "Switch cost",
    }

    # Context keys shown below the breakdown table (not reward components)
    _CONTEXT_LABELS: Dict[str, str] = {
        "matchup_score":        "  matchup_score",
        "threat_level":         "  threat_level",
        "damage_momentum":      "  momentum (raw EMA)",
        "ia_effectiveness":     "  move effectiveness",
        "consecutive_advantage":"  consec. advantage ctr",
        "switch_quality_delta": "  switch quality Δ",
        "is_true_stall":        "  true stall?",
    }

    @classmethod
    def explain(cls, breakdown: dict, threshold: float = 1e-7) -> str:
        """
        Return a formatted multi-line string of the reward breakdown.

        Parameters
        ----------
        breakdown : dict
            The dict returned in info["reward_breakdown"] from env.step().
        threshold : float
            Components with |value| < threshold are omitted from the table.
        """
        lines = [
            "",
            "  ┌─────────────────────────────────────────────────┐",
            "  │           REWARD BREAKDOWN (per step)           │",
            "  ├────────────────────────────────┬────────────────┤",
            f"  │ {'Component':<30} │ {'Value':>12}  │",
            "  ├────────────────────────────────┼────────────────┤",
        ]

        total = breakdown.get("reward", 0.0)
        for key, label in cls._LABELS.items():
            value = breakdown.get(key, 0.0)
            if abs(value) >= threshold:
                sign = "+" if value > 0 else ""
                lines.append(f"  │ {label:<30} │ {sign}{value:12.5f}  │")

        lines += [
            "  ├────────────────────────────────┼────────────────┤",
            f"  │ {'TOTAL':<30} │ {'+' if total > 0 else ''}{total:12.5f}  │",
            "  ├────────────────────────────────┴────────────────┤",
            "  │ Context                                         │",
            "  ├─────────────────────────────────────────────────┤",
        ]

        for key, label in cls._CONTEXT_LABELS.items():
            val = breakdown.get(key)
            if val is not None:
                if isinstance(val, bool):
                    lines.append(f"  │ {label:<30}  {str(val):<15} │")
                elif isinstance(val, float):
                    lines.append(f"  │ {label:<30}  {val:+.5f}        │")
                else:
                    lines.append(f"  │ {label:<30}  {val:<15} │")

        lines.append("  └─────────────────────────────────────────────────┘")
        return "\n".join(lines)

    @classmethod
    def print(cls, breakdown: dict, **kwargs) -> None:
        """Print the reward breakdown directly to stdout."""
        print(cls.explain(breakdown, **kwargs))

    @classmethod
    def summary_line(cls, breakdown: dict) -> str:
        """
        Return a single compact summary line for logging.

        Example
        -------
        "total=+0.453 | terminal=+1.00 | ko=+0.22 | dmg=+0.05 | matchup=+0.02"
        """
        parts = [f"total={breakdown.get('reward', 0.0):+.3f}"]
        for key, short in [
            ("terminal_bonus",    "terminal"),
            ("ko_reward",         "ko"),
            ("damage_reward",     "dmg"),
            ("matchup_shaping",   "matchup"),
            ("temporal_risk",     "threat"),
            ("momentum_reward",   "momentum"),
            ("smart_switch",      "switch"),
        ]:
            v = breakdown.get(key, 0.0)
            if abs(v) >= 1e-6:
                parts.append(f"{short}={v:+.3f}")
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  TUNING GUIDE  (printed when module is run directly)
# ─────────────────────────────────────────────────────────────────────────────

TUNING_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              REWARD FUNCTION TUNING GUIDE — Pokémon RL System              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SYMPTOM → DIAGNOSIS → FIX                                                  ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  Agent always uses the same move                                             ║
║    → move_quality signal too weak vs terminal reward                         ║
║    → Increase move_quality_k (0.02 → 0.04)                                  ║
║    → Increase move_immune_pen magnitude (−0.06 → −0.10)                     ║
║                                                                              ║
║  Agent never switches even in bad matchups                                   ║
║    → bad_stay_penalty not firing, or switch cost too high                    ║
║    → Reduce bad_stay_threshold (e.g. −0.15 → −0.10)                         ║
║    → Increase bad_stay_matchup_k / bad_stay_threat_k                        ║
║    → Reduce switch_cost (0.03 → 0.01)                                       ║
║    → Increase smart_switch_k (0.06 → 0.09)                                  ║
║                                                                              ║
║  Agent spams switches (switch every turn)                                    ║
║    → switch_cost too low or smart_switch_k too high                          ║
║    → Increase switch_cost (0.03 → 0.06)                                     ║
║    → Reduce switch_fatigue_n (8 → 4) to trigger fatigue sooner              ║
║                                                                              ║
║  Agent ends battles in 1–2 turns (burst play)                                ║
║    → anti_burst_penalty not sufficient                                       ║
║    → Increase anti_burst_penalty (0.15 → 0.25)                              ║
║    → Increase anti_burst_turns (4 → 6)                                      ║
║    → Reduce ko_bonus (0.22 → 0.15)                                           ║
║                                                                              ║
║  Agent is overly passive / stalls frequently                                 ║
║    → stall_penalty too weak, or survival_bonus masking progress             ║
║    → Increase stall_penalty (0.015 → 0.030)                                 ║
║    → Reduce survival_bonus (0.008 → 0.004)                                  ║
║                                                                              ║
║  Agent ignores type matchups                                                 ║
║    → matchup_k too low                                                       ║
║    → Increase matchup_k (0.035 → 0.060)                                     ║
║    → Increase bad_stay_matchup_k (0.020 → 0.035)                            ║
║                                                                              ║
║  Win rate plateau: agent wins ~50% and stops improving                       ║
║    → Opponent too strong (self-play); or reward variance too high            ║
║    → Lower learning_rate in PPO (3e-4 → 1e-4)                               ║
║    → Increase ent_coef to maintain exploration (0.02 → 0.05)                ║
║    → Add more random-baseline episodes to self-play curriculum               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KEY HYPERPARAMETERS (PPO-level, not reward)                                 ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  learning_rate  3e-4   Lower if training is unstable; higher for fast init   ║
║  n_steps        2048   Steps per env per update. Increase for longer memory  ║
║  batch_size     128    Mini-batch size. Larger = more stable gradients       ║
║  ent_coef       0.02   Entropy bonus. Increase to maintain exploration       ║
║  n_epochs       10     PPO epochs per update. 10 is standard                ║
║  gamma          0.99   Discount. High is correct for long-horizon battles    ║
║  gae_lambda     0.95   GAE bias-variance tradeoff. 0.95 is standard         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    import dataclasses

    print("\n=== DEFAULT REWARD WEIGHTS ===\n")
    for f in dataclasses.fields(DEFAULT_WEIGHTS):
        val = getattr(DEFAULT_WEIGHTS, f.name)
        print(f"  {f.name:<28} = {val}")

    print(TUNING_GUIDE)
