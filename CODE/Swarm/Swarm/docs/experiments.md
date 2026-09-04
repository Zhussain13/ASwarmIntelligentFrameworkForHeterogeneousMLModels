# Suggested Experiments for Evaluation

These experiments are designed to produce quantitative graphs suitable for an IEEE/Springer conference submission.

1) Accuracy vs. Latency Trade-off (Skip Threshold Sweep)
   - Goal: Measure object detection accuracy and per-frame latency as the
     similarity threshold varies (e.g., 0.70 to 0.95).
   - Procedure: Run the simulation on recorded video(s) with a ground-truth
     labeled subset. For each threshold, record:
       - % frames where inference was skipped
       - detection F1-score (consider using labels from high-capacity agent as reference)
       - average latency per frame (ms)
   - Expected plots: threshold vs. F1, threshold vs. average latency.

2) Energy Savings in Heterogeneous Swarm
   - Goal: Estimate battery/energy savings by skipping heavy inference.
   - Procedure: Profile execution time and/or power draw for heavy models on
     representative hardware (e.g., Jetson Nano, Raspberry Pi 4). Simulate
     a swarm with mixed-capacity agents and measure cumulative energy used
     over a fixed scenario (e.g., 10 minutes of footage).
   - Expected plots: energy consumed vs. number of agents, or energy saved vs. skip rate.

3) Robustness to Concept Drift & Memory TTL
   - Goal: Evaluate system behavior when the scene changes (new objects) and
     when packet TTL varies.
   - Procedure: Use video sequences where objects appear/disappear. Sweep TTL
     values (e.g., 5s, 30s, 120s) and measure false positive skips (skipping
     when new object present) and missed detections.
   - Expected plots: TTL vs. false-skip-rate, TTL vs. detection latency.

Notes:
- Use multiple datasets (indoor, outdoor, crowded scenes) and report mean ± std.
- For statistical rigor, run multiple seeds and report confidence intervals.
