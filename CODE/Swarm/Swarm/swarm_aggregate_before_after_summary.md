| metric                            | before_baseline   | after_implementation   | advantage                                  |
|:----------------------------------|:------------------|:-----------------------|:-------------------------------------------|
| total_decisions                   | 60                | 60                     | Context metric                             |
| heavy_inference_calls             | 60                | 3                      | 57 fewer heavy calls (95.00% reduction)    |
| skipped_calls                     | 0                 | 57                     | +57 calls skipped                          |
| skip_rate_percent                 | 0.00              | 95.00                  | +95.00 pp                                  |
| heavy_rate_percent                | 100.00            | 5.00                   | -95.00 pp                                  |
| relative_heavy_compute_factor     | 1.00x             | 0.05x                  | Approx. 20.00x fewer heavy inferences      |
| skipped_similarity_mean           | N/A               | 0.996608               | Higher is better confidence proxy          |
| skipped_similarity_std            | N/A               | 0.001942               | Lower indicates more consistency           |
| cross_agent_transfer_rate_percent | 0.00              | 0.00                   | 0/57 skipped decisions reused peer packets |