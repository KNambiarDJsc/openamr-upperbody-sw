# SPDX-License-Identifier: MIT
"""The #44 lift guard's safety properties, checked over a whole recorded run instead of at a few scripted moments.

No ROS here: the functions take plain time series (wall seconds, monotonic) so they can be unit-tested with synthetic data
(test_lift_guard_oracle.py). lift_guard_support.Recorder fills a Timeline from the running stack.

Properties (each returns a list of violations, empty when it holds):

  P1 no_motion_while_blocked      while the guard reports ERROR, lift_joint does not move (after `settle` s from the change)
  P2 no_motion_while_inactive     while lift_controller is not active, lift_joint does not move (after `settle` s)
  P3 arm_out_is_blocked           whenever an arm joint is outside the stowed tolerance (or unreadable) for longer than `reaction` s, the
                                  guard reports ERROR with that arm's code, and lift_controller is not active. This is the test's own oracle,
                                  computed from the joint states it sees, independent of the guard: a guard that allows the lift for a moment
                                  anywhere in the run is caught, not only where a case looks
  P4 status_contract              every status: level OK or ERROR only; OK iff no blocking codes; codes from the published set only
  P5 status_matches_controller    the status's lift_controller value agrees with the controller manager within `lag` s
  P6 allowed_means_active         a status of OK with lift_controller inactive (or ERROR with it active) does not persist beyond `lag` s

What these are not: they say nothing about the firmware interlock (CTL-017), and they only see what the test process sees (the guard may
have seen a message the test missed, and the other way round; the reaction windows absorb that).
"""
import math
from dataclasses import dataclass, field

OK, ERROR = 0, 2           # diagnostic_msgs/DiagnosticStatus levels the guard may use


@dataclass
class Timeline:
    lift: list = field(default_factory=list)          # (t, position)
    arms: list = field(default_factory=list)          # (t, {"left": deviation, "right": deviation}); deviation = inf if unreadable/NaN
    status: list = field(default_factory=list)        # (t, level, controller_value, (codes...))
    controller: list = field(default_factory=list)    # (t, "active" | "inactive" | ...) for lift_controller
    marks: list = field(default_factory=list)         # (t, label): case boundaries, for the report
    end: float = 0.0


def label_at(timeline, t):
    """The case that was running at `t`, for violation messages."""
    current = None
    for tm, label in timeline.marks:
        if tm <= t:
            current = label
    return current


def step_intervals(events, predicate, end):
    """[(a, b)] where `predicate(event)` holds, treating `events` [(t, ...)] as a step function that changes at each event."""
    out, start = [], None
    for ev in events:
        on = predicate(ev)
        if on and start is None:
            start = ev[0]
        elif not on and start is not None:
            out.append((start, ev[0]))
            start = None
    if start is not None and end > start:
        out.append((start, end))
    return out


def shrink(intervals, settle):
    return [(a + settle, b) for a, b in intervals if b > a + settle]


def lift_motion(lift, a, b):
    """Peak-to-peak lift_joint over samples in [a, b], or 0.0 if fewer than two."""
    vals = [p for t, p in lift if a <= t <= b and not math.isnan(p)]
    return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0


def _value_at(events, t, index=1, default=None):
    value = default
    for ev in events:
        if ev[0] <= t:
            value = ev[index]
        else:
            break
    return value


def no_motion_while_blocked(tl, settle, eps):
    blocked = shrink(step_intervals(tl.status, lambda e: e[1] != OK, tl.end), settle)
    return [{"property": "P1", "from": a, "to": b, "moved_m": m, "case": label_at(tl, a)}
            for a, b in blocked for m in [lift_motion(tl.lift, a, b)] if m > eps]


def no_motion_while_inactive(tl, settle, eps):
    inactive = shrink(step_intervals(tl.controller, lambda e: e[1] != "active", tl.end), settle)
    return [{"property": "P2", "from": a, "to": b, "moved_m": m, "case": label_at(tl, a)}
            for a, b in inactive for m in [lift_motion(tl.lift, a, b)] if m > eps]


def arm_out_is_blocked(tl, tolerance, reaction, sides=("left", "right")):
    """P3. The oracle's 'arm out' is deviation > tolerance * 1.05 (a 5 % band so that a joint resting exactly on the tolerance does not
    make the oracle and the guard disagree about a rounding)."""
    out = []
    for side in sides:
        code = f"ARM_NOT_STOWED_{side.upper()}"
        alt = (code, f"ARM_TRAJECTORY_ACTIVE_{side.upper()}", "ARM_STATE_MISSING", "ARM_STATE_STALE")
        for a, b in shrink(step_intervals(tl.arms, lambda e: e[1][side] > tolerance * 1.05, tl.end), reaction):
            if _value_at(tl.status, a, index=1) == OK:
                out.append({"property": "P3", "side": side, "at": a, "level": OK, "codes": [], "case": label_at(tl, a)})
            for t, level, ctl, codes in tl.status:
                if a <= t <= b and (level == OK or not any(c in codes for c in alt)):
                    out.append({"property": "P3", "side": side, "at": t, "level": level, "codes": list(codes), "case": label_at(tl, t)})
                    break
            # lift_controller may be active inside the interval only briefly: an outside activation the guard undoes within `reaction`.
            # Active when the interval starts (the guard did not deactivate within `reaction` of the arm leaving), or active for longer
            # than `reaction` anywhere inside it, is a violation.
            for c0, c1 in step_intervals(tl.controller, lambda e: e[1] == "active", tl.end):
                x, y = max(a, c0), min(b, c1)
                if y <= x:
                    continue
                if c0 < a or y - x > reaction:
                    out.append({"property": "P3", "side": side, "at": x, "controller": "active", "for_s": round(y - x, 3),
                                "case": label_at(tl, x)})
                    break
    return out


def status_contract(tl, known_codes):
    out = []
    for t, level, ctl, codes in tl.status:
        problems = []
        if level not in (OK, ERROR):
            problems.append(f"level {level} is neither OK nor ERROR")
        if (level == OK) != (len(codes) == 0):
            problems.append(f"level {level} with blocking codes {list(codes)}")
        unknown = [c for c in codes if c not in known_codes]
        if unknown:
            problems.append(f"unknown codes {unknown}")
        if ctl not in ("active", "inactive"):
            problems.append(f"lift_controller value {ctl!r}")
        if problems:
            out.append({"property": "P4", "at": t, "problems": problems, "case": label_at(tl, t)})
    return out


def status_matches_controller(tl, lag):
    """P5: sample the status's controller value at every status event; it must match the controller manager's state at some moment in
    [t - lag, t + lag]."""
    def norm(state):
        return None if state is None else ("active" if state == "active" else "inactive")

    out = []
    for t, level, ctl, codes in tl.status:
        if ctl not in ("active", "inactive"):
            continue
        window = {norm(_value_at(tl.controller, t - lag))} | {norm(s) for tc, s in tl.controller if t - lag <= tc <= t + lag}
        window.discard(None)
        if window and ctl not in window:
            out.append({"property": "P5", "at": t, "status_says": ctl, "manager_says": sorted(window), "case": label_at(tl, t)})
    return out


def allowed_means_active(tl, lag):
    """P6: OK must come with lift_controller active, ERROR with it inactive, apart from transitions shorter than `lag`."""
    def reached(t, want_active):
        """lift_controller was in the wanted state at t, or got there at some moment in (t, t + lag]."""
        def ok(state):
            return (state == "active") == want_active
        return ok(_value_at(tl.controller, t)) or any(t < tc <= t + lag and ok(s) for tc, s in tl.controller)

    out = []
    for t, level, ctl, codes in tl.status:
        if level == OK and not reached(t, True):
            out.append({"property": "P6", "at": t, "status": "OK", "controller": _value_at(tl.controller, t + lag), "case": label_at(tl, t)})
        if level == ERROR and not reached(t, False):
            out.append({"property": "P6", "at": t, "status": "ERROR", "controller": _value_at(tl.controller, t + lag), "case": label_at(tl, t)})
    return out


def evaluate(tl, *, tolerance, known_codes, settle, eps, reaction, lag):
    """All properties; {property: [violations]}."""
    return {
        "P1_no_motion_while_blocked": no_motion_while_blocked(tl, settle, eps),
        "P2_no_motion_while_inactive": no_motion_while_inactive(tl, settle, eps),
        "P3_arm_out_is_blocked": arm_out_is_blocked(tl, tolerance, reaction),
        "P4_status_contract": status_contract(tl, known_codes),
        "P5_status_matches_controller": status_matches_controller(tl, lag),
        "P6_allowed_means_active": allowed_means_active(tl, lag),
    }


def arm_deviation(positions, park, joints):
    """Largest |position - park| over `joints`; inf if a joint is missing or not finite (a NaN must never read as 'stowed')."""
    worst = 0.0
    for j in joints:
        p = positions.get(j)
        if p is None or not math.isfinite(p):
            return math.inf
        worst = max(worst, abs(p - park[j]))
    return worst


def first_crossing(series, predicate, after=0.0):
    """Time of the first (t, ...) sample at or after `after` where `predicate(sample)` holds, or None."""
    for s in series:
        if s[0] >= after and predicate(s):
            return s[0]
    return None
