from typing import Any, Dict, List, Optional, Tuple
import math

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env_action import RailEnvActions


DIR_DELTA = {
    0: (-1, 0),  # North
    1: (0, 1),   # East
    2: (1, 0),   # South
    3: (0, -1),  # West
}


def _tuple_pos(pos):
    if pos is None:
        return None
    try:
        return (int(pos[0]), int(pos[1]))
    except Exception:
        return None


def _next_cell(pos, direction):
    pos = _tuple_pos(pos)
    if pos is None or direction is None:
        return None
    dr, dc = DIR_DELTA[int(direction)]
    return (pos[0] + dr, pos[1] + dc)


def _safe_status_name(agent):
    try:
        return str(agent.status.name)
    except Exception:
        return str(getattr(agent, "status", ""))


def _safe_malfunction(agent) -> int:
    try:
        return int(agent.malfunction_data.get("malfunction", 0))
    except Exception:
        return 0


def _safe_speed(agent) -> float:
    try:
        return float(agent.speed_data.get("speed", 1.0))
    except Exception:
        return 1.0


def _safe_latest_arrival(agent):
    for name in ["latest_arrival", "arrival_time", "latest_arrival_time"]:
        v = getattr(agent, name, None)
        if v is not None:
            return v
    return None


def _safe_earliest_departure(agent):
    for name in ["earliest_departure", "earliest_departure_time", "departure_time"]:
        v = getattr(agent, name, None)
        if v is not None:
            return v
    return None


def _planning_position(agent):
    p = _tuple_pos(getattr(agent, "position", None))
    if p is not None:
        return p
    return _tuple_pos(getattr(agent, "initial_position", None))


def _relative_action(current_direction, new_direction):
    if current_direction is None or new_direction is None:
        return int(RailEnvActions.DO_NOTHING)
    cd = int(current_direction)
    nd = int(new_direction)
    diff = (nd - cd) % 4
    if diff == 0:
        return int(RailEnvActions.MOVE_FORWARD)
    if diff == 1:
        return int(RailEnvActions.MOVE_RIGHT)
    if diff == 3:
        return int(RailEnvActions.MOVE_LEFT)
    return int(RailEnvActions.STOP_MOVING)


class MyObservationBuilder(ObservationBuilder):
    """
    Centralized observation builder for a hybrid OR-style controller.

    It intentionally returns rich global information to every agent observation,
    because MyPolicy is centralized and uses act_many.
    """

    def __init__(
        self,
        max_first_actions: int = 4,
        lookahead_depth: int = 7,
    ):
        super().__init__()
        self.max_first_actions = int(max_first_actions)
        self.lookahead_depth = int(lookahead_depth)
        self._last_snapshot = None

    def reset(self):
        self._last_snapshot = None

    def _elapsed(self):
        return int(getattr(self.env, "_elapsed_steps", 0))

    def _grid_shape(self):
        try:
            return int(self.env.height), int(self.env.width)
        except Exception:
            return None

    def _in_bounds(self, pos):
        pos = _tuple_pos(pos)
        if pos is None:
            return False
        shape = self._grid_shape()
        if shape is None:
            return True
        h, w = shape
        return 0 <= pos[0] < h and 0 <= pos[1] < w

    def _distance(self, handle, pos, direction, target):
        pos = _tuple_pos(pos)
        target = _tuple_pos(target)
        if pos is None:
            return 10**9

        try:
            dm = self.env.distance_map.get()
            d = dm[int(handle), int(pos[0]), int(pos[1]), int(direction)]
            d = float(d)
            if not math.isnan(d) and not math.isinf(d):
                return d
        except Exception:
            pass

        if target is not None:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

        return 10**9

    def _transitions(self, pos, direction):
        pos = _tuple_pos(pos)
        if pos is None or direction is None:
            return []
        try:
            transitions = self.env.rail.get_transitions(pos[0], pos[1], int(direction))
            return [int(i) for i, allowed in enumerate(transitions) if bool(allowed)]
        except Exception:
            return []

    def _greedy_rollout_after_first(self, handle, pos, direction, target, depth):
        """
        Greedy distance-map path from a candidate next state.
        Returns [(cell, direction, distance), ...].
        This is not meant as a perfect route, only as a reservation/lookahead signature.
        """
        path = []
        cur_pos = _tuple_pos(pos)
        cur_dir = int(direction) if direction is not None else None
        target = _tuple_pos(target)

        seen = set()

        for _ in range(int(depth)):
            if cur_pos is None or cur_dir is None:
                break
            if not self._in_bounds(cur_pos):
                break

            key = (cur_pos, cur_dir)
            if key in seen:
                break
            seen.add(key)

            dist = self._distance(handle, cur_pos, cur_dir, target)
            path.append((cur_pos, cur_dir, float(dist)))

            if target is not None and cur_pos == target:
                break

            next_dirs = self._transitions(cur_pos, cur_dir)
            if not next_dirs:
                break

            candidates = []
            for nd in next_dirs:
                np = _next_cell(cur_pos, nd)
                if not self._in_bounds(np):
                    continue
                d = self._distance(handle, np, nd, target)
                turn_penalty = 0.0 if nd == cur_dir else 0.15
                candidates.append((d + turn_penalty, d, np, nd))

            if not candidates:
                break

            candidates.sort(key=lambda x: (x[0], x[1], x[3]))
            _, _, cur_pos, cur_dir = candidates[0]

        return path

    def _first_move_candidates(self, handle, agent, occupied_now):
        pos = _planning_position(agent)
        true_pos = _tuple_pos(getattr(agent, "position", None))
        direction = getattr(agent, "direction", None)
        target = _tuple_pos(getattr(agent, "target", None))

        if pos is None or direction is None:
            return []

        base_d = self._distance(handle, pos, int(direction), target)

        candidates = []
        for nd in self._transitions(pos, int(direction)):
            next_pos = _next_cell(pos, nd)
            if not self._in_bounds(next_pos):
                continue

            action = _relative_action(int(direction), int(nd))
            if action == int(RailEnvActions.STOP_MOVING):
                continue

            d1 = self._distance(handle, next_pos, nd, target)
            rollout = self._greedy_rollout_after_first(
                handle=handle,
                pos=next_pos,
                direction=nd,
                target=target,
                depth=self.lookahead_depth,
            )

            # Heuristic cost:
            #   distance dominates
            #   turning penalty small
            #   moving into an occupied cell is not forbidden here, but marked
            #   improvement reward if the first move decreases distance
            turn_penalty = 0.0 if int(nd) == int(direction) else 0.20
            occupied_penalty = 8.0 if next_pos in occupied_now and next_pos != true_pos else 0.0
            improvement_bonus = -0.5 if d1 < base_d else 0.5

            cost = float(d1) + turn_penalty + occupied_penalty + improvement_bonus

            candidates.append({
                "action": int(action),
                "new_direction": int(nd),
                "next_position": next_pos,
                "distance_after": float(d1),
                "base_distance": float(base_d),
                "cost": float(cost),
                "rollout": [
                    {
                        "position": p,
                        "direction": int(di),
                        "distance": float(dd),
                    }
                    for p, di, dd in rollout
                ],
            })

        candidates.sort(key=lambda c: (c["cost"], c["distance_after"], c["action"]))
        return candidates[: self.max_first_actions]

    def _waypoints_like(self, agent):
        """
        Attempts to expose waypoint/stop information if present in this Flatland version.
        The policy can ignore it if unavailable.
        """
        keys = [
            "waypoints",
            "schedule",
            "waypoint_times",
            "intermediate_targets",
            "stops",
            "scheduled_stops",
        ]
        out = {}
        for k in keys:
            try:
                v = getattr(agent, k, None)
                if v is not None:
                    out[k] = str(v)
            except Exception:
                pass
        return out

    def _build_snapshot(self, handles=None):
        if handles is None:
            handles = list(range(len(self.env.agents)))

        elapsed = self._elapsed()

        occupied_now = {}
        for i, a in enumerate(self.env.agents):
            p = _tuple_pos(getattr(a, "position", None))
            if p is not None:
                occupied_now[p] = int(i)

        agents = {}

        for handle in handles:
            a = self.env.agents[int(handle)]

            pos = _planning_position(a)
            true_pos = _tuple_pos(getattr(a, "position", None))
            initial_pos = _tuple_pos(getattr(a, "initial_position", None))
            target = _tuple_pos(getattr(a, "target", None))
            direction = getattr(a, "direction", None)
            direction = int(direction) if direction is not None else None

            status = _safe_status_name(a)
            malfunction = _safe_malfunction(a)
            speed = _safe_speed(a)
            latest_arrival = _safe_latest_arrival(a)
            earliest_departure = _safe_earliest_departure(a)

            base_distance = self._distance(
                int(handle),
                pos,
                direction if direction is not None else 0,
                target,
            )

            slack = None
            if latest_arrival is not None and base_distance < 10**8:
                try:
                    # Speed is accounted coarsely; exact Flatland fractional speed dynamics are handled by env.
                    travel_estimate = base_distance / max(float(speed), 1e-6)
                    slack = float(latest_arrival) - float(elapsed) - float(travel_estimate)
                except Exception:
                    slack = None

            candidates = self._first_move_candidates(
                int(handle),
                a,
                occupied_now=set(occupied_now.keys()),
            )

            agents[int(handle)] = {
                "handle": int(handle),
                "status": status,
                "position": true_pos,
                "planning_position": pos,
                "initial_position": initial_pos,
                "target": target,
                "direction": direction,
                "malfunction": int(malfunction),
                "speed": float(speed),
                "latest_arrival": latest_arrival,
                "earliest_departure": earliest_departure,
                "base_distance": float(base_distance),
                "slack": slack,
                "candidates": candidates,
                "extra_schedule_info": self._waypoints_like(a),
            }

        snapshot = {
            "time": int(elapsed),
            "handles": [int(h) for h in handles],
            "occupied_now": {str(k): int(v) for k, v in occupied_now.items()},
            "occupied_cells": list(occupied_now.keys()),
            "agents": agents,
            "grid_shape": self._grid_shape(),
        }

        self._last_snapshot = snapshot
        return snapshot

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Any]:
        snapshot = self._build_snapshot(handles)
        return {
            int(h): {
                "self": snapshot["agents"][int(h)],
                "global": snapshot,
            }
            for h in snapshot["handles"]
        }

    def get(self, handle: int = 0):
        snapshot = self._build_snapshot([int(handle)])
        return {
            "self": snapshot["agents"][int(handle)],
            "global": snapshot,
        }
