# SPDX-License-Identifier: MIT
"""The #44 lift guard's safety properties, checked over a whole recorded run instead of at a few scripted moments.

No ROS here: the functions take plain time series (wall seconds, monotonic) so they can be unit-tested with synthetic data
(test_lift_guard_oracle.py). lift_guard_support.Recorder fills a Timeline from the running stack.

Properties (each returns a list of violations, empty when it holds):

  P1 no_motion_while_blocked      while the guard reports ERROR (always "blocked", #44 decision of 10 Nov), lift_joint does not move
                                  (after `settle` s from the change)
  P2 no_motion_while_inactive     while lift_controller is not active, lift_joint does not move (after `settle` s)
                                  P1 and P2 also need to SEE the lift: a stretch of such an interval in which lift_joint was not sampled
                                  for more than OBSERVE_GAP_S is itself a violation ("unobserved"), not a pass, unless the test declared it
                                  a blackout (Timeline.blackouts: it stopped the joint states on purpose)
  P3 arm_out_is_blocked           whenever an arm joint is outside the stowed tolerance (or unreadable) for longer than `reaction` s, the
                                  guard reports ERROR with that arm's code (OK or WARN is a breach), and lift_controller is not active. This
                                  is the test's own oracle,
                                  computed from the joint states it sees, independent of the guard: a guard that allows the lift for a moment
                                  anywhere in the run is caught, not only where a case looks
  P4 status_contract              the agreed contract (#44, 10 Nov): OK = nothing blocks and lift_controller active; WARN = nothing blocks,
                                  lift_controller "activating" (switching on); ERROR = blocked, with at least one code; codes from the
                                  published set only; lift_controller one of active, inactive, activating
  P5 status_matches_controller    the status's lift_controller value agrees with the controller manager within `lag` s ("activating"
                                  counts as not active yet)
  P6 allowed_means_active         OK with lift_controller not active, or ERROR with it active, does not persist beyond `lag` s (the
                                  status read as a step function: a status in flight across a switch is not a breach)
  P7 switching_on_is_brief        WARN ("activating") is transitional: each WARN spell ends within `warn_max` s (OK, or ERROR with
                                  SWITCH_FAILED). Spells before lift_controller is first seen at all (still being loaded) are not counted

What these are not: they say nothing about the firmware interlock (CTL-017), and they only see what the test process sees (the guard may
have seen a message the test missed, and the other way round; the reaction windows absorb that).
"""
import math
from dataclasses import dataclass, field

OK, WARN, ERROR = 0, 1, 2  # diagnostic_msgs/DiagnosticStatus levels the guard may use
CONTROLLER_VALUES = ("active", "inactive", "activating")


@dataclass
class Timeline:
    lift: list = field(default_factory=list)          # (t, position)
    arms: list = field(default_factory=list)          # (t, {"left": deviation, "right": deviation}); deviation = inf if unreadable/NaN
    status: list = field(default_factory=list)        # (t, level, controller_value, (codes...))
    controller: list = field(default_factory=list)    # (t, "active" | "inactive" | ...) for lift_controller
    marks: list = field(default_factory=list)         # (t, label): case boundaries, for the report
    end: float = 0.0
    blackouts: list = field(default_factory=list)     # (t0, t1): spans in which the test itself stopped the joint states
    status_end: float = None                          # when the status stream ended (the guard's exit, L03); None: at `end`


def _status_end(tl):
    """A status holds until the next one, the last until the guard's status stream ended: a dead guard says nothing."""
    return tl.end if tl.status_end is None else min(tl.status_end, tl.end)


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
    """Peak-to-peak lift_joint over samples in [a, b], or 0.0 if fewer than two. Whether the lift was SEEN is unobserved()'s question:
    this alone would call an unobserved interval 'no motion'."""
    vals = [p for t, p in lift if a <= t <= b and not math.isnan(p)]
    return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0


OBSERVE_GAP_S = 2.0      # the longest stretch inside a P1/P2 interval in which lift_joint may go unsampled (mock: 100 Hz; Gazebo: slower)


def unobserved(tl, a, b, max_gap=OBSERVE_GAP_S):
    """[(from, to, gap_s)]: the stretches of [a, b], outside the declared blackouts, whose longest gap between lift_joint samples (a NaN
    is not a sample; the stretch's own ends count) is more than `max_gap`. With max_gap=0 every stretch is returned: its coverage.
    Before the FIRST lift sample there is nothing to see yet (the guard says ERROR while the stack is still coming up): a stretch starts
    no earlier than that sample. With no sample at all nothing is clipped, so a lift that was never seen is reported."""
    spans = [(a, b)]
    for b0, b1 in sorted(tl.blackouts):
        pieces = []
        for x, y in spans:
            if b1 <= x or b0 >= y:
                pieces.append((x, y))
            else:
                pieces += [(x, min(y, b0)), (max(x, b1), y)]
        spans = [(x, y) for x, y in pieces if y > x]
    seen = sorted(t for t, p in tl.lift if not math.isnan(p))
    out = []
    for x, y in spans:
        if seen:
            x = max(x, seen[0])
        if y <= x:
            continue
        edges = [x] + [t for t in seen if x <= t <= y] + [y]
        gap = max(q - r for r, q in zip(edges, edges[1:]))
        if gap > max_gap:
            out.append((x, y, round(gap, 3)))
    return out


def _lift_violations(tl, prop, intervals, eps):
    out = []
    for a, b in intervals:
        m = lift_motion(tl.lift, a, b)
        if m > eps:
            out.append({"property": prop, "from": a, "to": b, "moved_m": m, "case": label_at(tl, a)})
        out += [{"property": prop, "unobserved": True, "from": x, "to": y, "gap_s": g, "case": label_at(tl, x)}
                for x, y, g in unobserved(tl, a, b)]
    return out


def _value_at(events, t, index=1, default=None):
    value = default
    for ev in events:
        if ev[0] <= t:
            value = ev[index]
        else:
            break
    return value


def no_motion_while_blocked(tl, settle, eps):
    # ERROR always means blocked (#44 decision of 10 Nov). WARN ("activating") is not blocked; P4 makes sure it carries no code.
    blocked = shrink(step_intervals(tl.status, lambda e: e[1] == ERROR, _status_end(tl)), settle)
    return _lift_violations(tl, "P1", blocked, eps)


def no_motion_while_inactive(tl, settle, eps):
    inactive = shrink(step_intervals(tl.controller, lambda e: e[1] != "active", tl.end), settle)
    return _lift_violations(tl, "P2", inactive, eps)


def arm_out_is_blocked(tl, tolerance, reaction, sides=("left", "right")):
    """P3. The oracle's 'arm out' is deviation > tolerance * 1.05 (a 5 % band so that a joint resting exactly on the tolerance does not
    make the oracle and the guard disagree about a rounding)."""
    out = []
    for side in sides:
        code = f"ARM_NOT_STOWED_{side.upper()}"
        alt = (code, f"ARM_TRAJECTORY_ACTIVE_{side.upper()}", "ARM_STATE_MISSING", "ARM_STATE_STALE")
        for a, b in shrink(step_intervals(tl.arms, lambda e: e[1][side] > tolerance * 1.05, tl.end), reaction):
            if _value_at(tl.status, a, index=1) in (OK, WARN):
                out.append({"property": "P3", "side": side, "at": a, "level": _value_at(tl.status, a, index=1), "codes": [],
                            "case": label_at(tl, a)})
            for t, level, ctl, codes in tl.status:
                if a <= t <= b and (level != ERROR or not any(c in codes for c in alt)):
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
    """P4, the contract agreed on #44 (10 Nov): OK = nothing blocks and lift_controller active; WARN = nothing blocks, lift_controller
    "activating"; ERROR = blocked, with at least one code."""
    out = []
    for t, level, ctl, codes in tl.status:
        problems = []
        if level not in (OK, WARN, ERROR):
            problems.append(f"level {level} is not OK, WARN or ERROR")
        if level in (OK, WARN) and codes:
            problems.append(f"level {level} with blocking codes {list(codes)}")
        if level == OK and ctl != "active":
            problems.append(f"OK while lift_controller is reported {ctl!r}")
        if level == WARN and ctl != "activating":
            problems.append(f"WARN while lift_controller is reported {ctl!r} (WARN is the 'activating' state)")
        if level == ERROR and not codes:
            problems.append("ERROR with no blocking code (ERROR always means blocked, and blocking names the cause)")
        if level == ERROR and ctl == "activating":
            problems.append("ERROR while lift_controller is reported 'activating'")
        unknown = [c for c in codes if c not in known_codes]
        if unknown:
            problems.append(f"unknown codes {unknown}")
        if ctl not in CONTROLLER_VALUES:
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
        if ctl not in CONTROLLER_VALUES:
            continue
        ctl = "inactive" if ctl == "activating" else ctl     # switching on: not active yet
        window = {norm(_value_at(tl.controller, t - lag))} | {norm(s) for tc, s in tl.controller if t - lag <= tc <= t + lag}
        window.discard(None)
        if window and ctl not in window:
            out.append({"property": "P5", "at": t, "status_says": ctl, "manager_says": sorted(window), "case": label_at(tl, t)})
    return out


def allowed_means_active(tl, lag):
    """P6: OK with lift_controller not active, or ERROR with it active, does not persist beyond `lag` s. The status is read as a step
    function, like the controller's state: each status holds until the next one, the last until the status stream ends. So a status that
    was already in flight when the controller switched is not a breach (12 Nov, first run on the integrated stack: an OK published while
    the controller was active arrived 0-7 ms after the deactivation event and 7 ms before the guard's ERROR; the per-status check used until
    then looked only forward and flagged it), and a guard that keeps saying OK after its controller went inactive is. WARN (switching on)
    asks nothing of the controller here: an activation the guard has not seen yet is not a breach, and P7 bounds how long WARN lasts."""
    end = _status_end(tl)
    status = sorted(tl.status, key=lambda e: e[0])
    controller = sorted(tl.controller, key=lambda e: e[0])
    points = sorted({e[0] for e in status if e[0] <= end} | {e[0] for e in controller if e[0] <= end})
    out = []
    for level, name, wants_active in ((OK, "OK", True), (ERROR, "ERROR", False)):
        since = None
        for t in points + [end]:
            on = t < end and _value_at(status, t, index=1) == level and (_value_at(controller, t) == "active") != wants_active
            if on and since is None:
                since = t
            elif not on and since is not None:
                if t - since > lag:
                    out.append({"property": "P6", "at": since, "for_s": round(t - since, 3), "status": name,
                                "controller": _value_at(controller, since), "case": label_at(tl, since)})
                since = None
    return sorted(out, key=lambda v: v["at"])


def switching_on_is_brief(tl, warn_max):
    """P7: every WARN spell (consecutive WARN statuses) ends within `warn_max` s. Clipped to start no earlier than the first observation of
    lift_controller: before the controller manager has even loaded it, 'activating' can legitimately last as long as start-up does."""
    first_seen = tl.controller[0][0] if tl.controller else None
    out = []
    if first_seen is None:
        return out
    for a, b in step_intervals(tl.status, lambda e: e[1] == WARN, _status_end(tl)):
        a = max(a, first_seen)
        if b - a > warn_max:
            out.append({"property": "P7", "from": a, "to": b, "for_s": round(b - a, 3), "case": label_at(tl, a)})
    return out


def evaluate(tl, *, tolerance, known_codes, settle, eps, reaction, lag, warn_max=10.0):
    """All properties; {property: [violations]}."""
    return {
        "P1_no_motion_while_blocked": no_motion_while_blocked(tl, settle, eps),
        "P2_no_motion_while_inactive": no_motion_while_inactive(tl, settle, eps),
        "P3_arm_out_is_blocked": arm_out_is_blocked(tl, tolerance, reaction),
        "P4_status_contract": status_contract(tl, known_codes),
        "P5_status_matches_controller": status_matches_controller(tl, lag),
        "P6_allowed_means_active": allowed_means_active(tl, lag),
        "P7_switching_on_is_brief": switching_on_is_brief(tl, warn_max),
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


def stowed_deviation(positions, poses, joints):
    """Largest |position - pose| over `joints` for the CLOSEST of the stowed `poses`: the guard's own rule per arm (every joint within the
    tolerance of ANY stowed state), so the oracle follows stowed_states and does not assume park is the only safe pose (robot rule: stowed
    OR carry). inf if a joint is missing or not finite, or if there is no pose to compare with."""
    return min((arm_deviation(positions, pose, joints) for pose in poses), default=math.inf)


def first_not_active_after(controller, t):
    """When lift_controller was first seen not active at or after `t`: `t` itself if its last known state at `t` was not active (a step
    function, like the status), else the first later sample that is; None if it never was, or if nothing was known."""
    state = None
    for s in controller:
        if s[0] <= t:
            state = s[1]
        elif state is not None and state != "active":
            break
        elif s[1] != "active":
            return s[0]
    return t if state is not None and state != "active" else None


def first_crossing(series, predicate, after=0.0):
    """Time of the first (t, ...) sample at or after `after` where `predicate(sample)` holds, or None."""
    for s in series:
        if s[0] >= after and predicate(s):
            return s[0]
    return None
