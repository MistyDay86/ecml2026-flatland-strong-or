from typing import Any, Dict, List, Tuple
from collections import defaultdict

from flatland.envs.RailEnvPolicy import RailEnvPolicy
from flatland.envs.rail_env_action import RailEnvActions



def _action_value(x):
    try:
        return int(x.value)
    except Exception:
        try:
            return int(x)
        except Exception:
            return int(RailEnvActions.DO_NOTHING.value)


def _to_action(x):
    try:
        return RailEnvActions(x)
    except Exception:
        try:
            return RailEnvActions.from_value(int(x))
        except Exception:
            return RailEnvActions.DO_NOTHING


def _pos_tuple(x):
    if x is None:
        return None
    try:
        return (int(x[0]), int(x[1]))
    except Exception:
        return None


def _status_done(status):
    s = str(status).upper()
    return "DONE" in s or "REMOVED" in s


def _status_active(status):
    s = str(status).upper()
    return ("ACTIVE" in s) or ("MOVING" in s) or ("MALFUNCTION" in s)


def _status_not_started(status):
    s = str(status).upper()
    return ("READY" in s) or ("WAIT" in s) or ("OFF" in s)


class MyPolicy(RailEnvPolicy):
    """
    Strong immediate hybrid OR policy.

    Main mechanisms:
      - centralized act_many
      - dynamic urgency ordering
      - short-horizon temporal reservation table
      - path rollout conflict scoring
      - stall/deadlock memory
      - safe fallback action
    """

    def __init__(
        self,
        horizon: int = 7,
        deadlock_boost_after: int = 8,
        max_stall: int = 18,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.deadlock_boost_after = int(deadlock_boost_after)
        self.max_stall = int(max_stall)

        self.t = 0
        self.last_pos = {}
        self.stall_count = defaultdict(int)
        self.last_action = {}
        self.last_distance = {}
        self.oscillation_count = defaultdict(int)

    def _normalize_obs(self, handles, observations):
        if isinstance(observations, dict):
            return observations

        out = {}
        for idx, h in enumerate(handles):
            try:
                out[int(h)] = observations[int(h)]
            except Exception:
                try:
                    out[int(h)] = observations[idx]
                except Exception:
                    out[int(h)] = None
        return out

    def _extract_snapshot(self, handles, observations):
        obs_by_handle = self._normalize_obs(handles, observations)

        for h in handles:
            obs = obs_by_handle.get(int(h), None)
            if isinstance(obs, dict) and "global" in obs:
                return obs["global"], obs_by_handle

        return None, obs_by_handle

    def _update_memory(self, agents):
        for h, info in agents.items():
            h = int(h)
            pos = _pos_tuple(info.get("position", None))
            planning_pos = _pos_tuple(info.get("planning_position", None))
            d = float(info.get("base_distance", 10**9))

            memory_pos = pos if pos is not None else planning_pos

            if memory_pos is not None and self.last_pos.get(h, None) == memory_pos:
                self.stall_count[h] += 1
            else:
                self.stall_count[h] = 0

            # Oscillation / non-progress memory.
            if h in self.last_distance:
                if d >= self.last_distance[h] - 1e-9:
                    self.oscillation_count[h] += 1
                else:
                    self.oscillation_count[h] = 0

            self.last_pos[h] = memory_pos
            self.last_distance[h] = d

    def _priority(self, h, info):
        status = str(info.get("status", ""))
        slack = info.get("slack", None)
        dist = float(info.get("base_distance", 10**8))
        malfunction = int(info.get("malfunction", 0))
        stall = int(self.stall_count[int(h)])
        osc = int(self.oscillation_count[int(h)])

        if _status_done(status):
            return (10**9, 10**9, int(h))

        # Malfunctioning trains cannot move; deprioritize but keep safe stop.
        if malfunction > 0:
            return (10**8, 10**8, int(h))

        if slack is None:
            slack_score = 10000.0
        else:
            slack_score = float(slack)

        # Negative slack is urgent.
        # Stalled agents become urgent to break deadlock.
        # Active agents are prioritized over not-yet-started ones.
        active_bonus = -250.0 if _status_active(status) else 0.0
        not_started_penalty = 350.0 if _status_not_started(status) else 0.0
        deadlock_bonus = -120.0 * max(0, stall - self.deadlock_boost_after)
        oscillation_bonus = -40.0 * max(0, osc - 5)

        # Closer trains are cheaper to finish; but urgency dominates.
        score = (
            7.0 * slack_score
            + 0.35 * dist
            + active_bonus
            + not_started_penalty
            + deadlock_bonus
            + oscillation_bonus
        )

        return (score, dist, int(h))

    def _candidate_conflict_cost(
        self,
        h,
        candidate,
        reservation,
        occupied_now,
        planned_leave,
        info,
    ):
        """
        Compute soft/hard conflict cost for a candidate rollout.

        Hard rejection only for immediately dangerous moves.
        Future conflicts are soft penalties because an overly conservative policy deadlocks.
        """
        cost = float(candidate.get("cost", 0.0))
        next_pos = _pos_tuple(candidate.get("next_position", None))
        current_pos = _pos_tuple(info.get("position", None))

        if next_pos is None:
            return None

        # Immediate cell already occupied:
        # Allowed only if the occupying agent is already planned to leave that cell.
        if next_pos in occupied_now and next_pos != current_pos:
            blocker = occupied_now.get(next_pos)
            if blocker not in planned_leave.get(next_pos, set()):
                # Hard reject unless this train has been badly stuck; then allow as huge penalty
                # only if no safer candidates exist.
                if self.stall_count[int(h)] < self.max_stall:
                    return None
                cost += 500.0

        # Same next cell collision.
        if next_pos in reservation.get(1, set()):
            return None

        # Penalize future reservation overlap.
        rollout = candidate.get("rollout", [])
        for tau, step in enumerate(rollout[: self.horizon], start=1):
            p = _pos_tuple(step.get("position", None))
            if p is None:
                continue

            if p in reservation.get(tau, set()):
                cost += 70.0 / max(1, tau)

            # Penalize being too close to occupied cells in first steps.
            if tau <= 2 and p in occupied_now and p != current_pos:
                cost += 15.0

        # Stalled agents should be willing to take slightly less optimal routes.
        stall = self.stall_count[int(h)]
        if stall > self.deadlock_boost_after:
            cost -= min(60.0, 3.5 * (stall - self.deadlock_boost_after))

        return float(cost)

    def _reserve_candidate(self, h, info, candidate, reservation, planned_leave):
        current_pos = _pos_tuple(info.get("position", None))
        next_pos = _pos_tuple(candidate.get("next_position", None))

        if current_pos is not None and next_pos is not None and current_pos != next_pos:
            planned_leave[current_pos].add(int(h))

        rollout = candidate.get("rollout", [])
        for tau, step in enumerate(rollout[: self.horizon], start=1):
            p = _pos_tuple(step.get("position", None))
            if p is None:
                continue
            reservation[tau].add(p)

        if next_pos is not None:
            reservation[1].add(next_pos)

    def _safe_wait_action(self, info):
        status = str(info.get("status", ""))
        if _status_not_started(status):
            return RailEnvActions.DO_NOTHING
        return RailEnvActions.STOP_MOVING

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        self.t += 1

        snapshot, obs_by_handle = self._extract_snapshot(handles, observations)

        if snapshot is None:
            # Extremely defensive fallback: keep trains from crashing randomly.
            return {int(h): RailEnvActions.DO_NOTHING for h in handles}

        agents = snapshot.get("agents", {})
        agents = {int(k): v for k, v in agents.items()}

        self._update_memory(agents)

        occupied_now = {}
        for p in snapshot.get("occupied_cells", []):
            pp = _pos_tuple(p)
            if pp is not None:
                occupied_now[pp] = None

        # Better occupied mapping from agent states.
        for h, info in agents.items():
            p = _pos_tuple(info.get("position", None))
            if p is not None:
                occupied_now[p] = int(h)

        actions = {int(h): RailEnvActions.DO_NOTHING for h in handles}

        reservation = defaultdict(set)       # tau -> set(cells)
        planned_leave = defaultdict(set)     # cell -> set(agent_handles)

        # Agents ordered by dynamic urgency.
        ordered = sorted(
            [int(h) for h in handles],
            key=lambda h: self._priority(int(h), agents.get(int(h), {}))
        )

        for h in ordered:
            info = agents.get(int(h), {})
            status = str(info.get("status", ""))

            if _status_done(status):
                actions[int(h)] = RailEnvActions.DO_NOTHING
                continue

            if int(info.get("malfunction", 0)) > 0:
                actions[int(h)] = RailEnvActions.STOP_MOVING
                continue

            candidates = info.get("candidates", [])

            best = None
            best_cost = None

            for cand in candidates:
                cc = self._candidate_conflict_cost(
                    h=int(h),
                    candidate=cand,
                    reservation=reservation,
                    occupied_now=occupied_now,
                    planned_leave=planned_leave,
                    info=info,
                )

                if cc is None:
                    continue

                if best is None or cc < best_cost:
                    best = cand
                    best_cost = cc

            if best is None:
                actions[int(h)] = self._safe_wait_action(info)
                self.last_action[int(h)] = _action_value(actions[int(h)])
                continue

            self._reserve_candidate(
                h=int(h),
                info=info,
                candidate=best,
                reservation=reservation,
                planned_leave=planned_leave,
            )

            action = _to_action(best.get("action", RailEnvActions.MOVE_FORWARD.value))
            actions[int(h)] = action
            self.last_action[int(h)] = _action_value(action)

        return actions

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        # Official runners should call act_many. This fallback is intentionally conservative.
        if isinstance(observation, dict):
            info = observation.get("self", {})
            candidates = info.get("candidates", [])
            if candidates:
                return _to_action(candidates[0].get("action", RailEnvActions.MOVE_FORWARD.value))
        return RailEnvActions.DO_NOTHING
