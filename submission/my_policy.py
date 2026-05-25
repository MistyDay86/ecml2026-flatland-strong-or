from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy


class MyPolicy(DeadLockAvoidancePolicy):
    def __init__(self):
        super().__init__(
            min_free_cell=1,
            show_debug_plot=False,
            count_num_opp_agents_towards_min_free_cell=True,
            use_switches_heuristic=True,

            # Miglioria rispetto alla configurazione base:
            # evita che due treni entrino simultaneamente in conflitto già dalla partenza.
            use_entering_prevention=True,

            # Prova alternative quando ci sono fermate/intermediate e un treno resta bloccato.
            # Valore piccolo: non esplode il costo computazionale.
            use_alternative_at_first_intermediate_and_then_always_first_strategy=3,

            # Se un agente è bloccato per troppi step, prova a saltare il prossimo waypoint flessibile.
            # Utile negli scenari con intermediate stops/disruption.
            drop_next_threshold=12,

            # Limite ragionevole per evitare path search troppo lunga.
            k_shortest_path_cutoff=3000,

            seed=42,
            verbose=False,
            audit=False,
        )
