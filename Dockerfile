FROM ghcr.io/flatland-association/flatland-baselines-deadlock-avoidance-heuristic:v4.2.5

COPY submission/ submission/

ENV POLICY=submission.my_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilder
