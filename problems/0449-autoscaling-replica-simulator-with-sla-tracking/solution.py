def autoscale_simulate(
    rps_series: list,
    capacity_per_replica: int,
    min_replicas: int,
    max_replicas: int,
    scale_up_threshold: float,
    scale_down_threshold: float,
    cooldown_steps: int
) -> dict:
    replicas = min_replicas
    cooldown_remaining = 0

    total_sla_violations = 0
    total_dropped_requests = 0
    utilization_sum = 0.0
    max_replicas_used = replicas

    for rps in rps_series:
        capacity = replicas * capacity_per_replica
        utilization = rps / capacity
        utilization_sum += utilization

        if rps > capacity:
            total_sla_violations += 1
            total_dropped_requests += (rps - capacity)

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        else:
            new_replicas = replicas

            if utilization > scale_up_threshold:
                new_replicas = min(replicas + 1, max_replicas)
            elif utilization < scale_down_threshold:
                new_replicas = max(repl