from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


class MyPolicy(DeadLockAvoidancePolicy):
    def __init__(self):
        super().__init__(
            min_free_cell=1,
            show_debug_plot=False,
            count_num_opp_agents_towards_min_free_cell=True,
            use_switches_heuristic=True,

            # v5 FAST:
            # disattiviamo tutte le parti costose che hanno mandato v4 in timeout.
            use_entering_prevention=False,
            use_alternative_at_first_intermediate_and_then_always_first_strategy=None,
            drop_next_threshold=None,
            k_shortest_path_cutoff=None,

            seed=42,
            verbose=False,
            audit=False,
        )
