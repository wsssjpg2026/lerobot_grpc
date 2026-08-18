"""Follower response-characterization suite (stage B-1).

Three behavioural layers over one shared harness, all against the 8-feature
pose-delta schema (``lerobot_robot_grpc.pose_delta_schema``):

- :mod:`.injectors` — synthetic delta sequences (pure functions, data-driven);
- :mod:`.backends`  — sim (in-process MuJoCo servicer on a real gRPC server,
  this round) and real (interface only, human-triggered later) backends;
- :mod:`.harness`   — 30 Hz action loop + 50 Hz observation sampler per sequence;
- :mod:`.metrics`   — tracking / lag / smoothness / stream metrics;
- :mod:`.report`    — JSON + CSV + Markdown baseline report under outputs/.

Tests assert the response is *sane* (no divergence, right direction,
converges); precise numbers land in the report as baseline data for B-2/B-3
human review — thresholds are placeholders there, not pass/fail gates.
"""
