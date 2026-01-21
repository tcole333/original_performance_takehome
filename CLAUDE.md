# Performance Take-Home Project

## Quick Reference
- **Current cycles:** Check with `uv run python perf_takehome.py Tests.test_kernel_cycles`
- **Full tests:** `uv run python tests/submission_tests.py`
- **Trace visualization:** `uv run python perf_takehome.py Tests.test_kernel_trace` then open Perfetto

## Benchmarks (from Readme.md)
- 2,164: Opus 4 (many hours)
- 1,790: Opus 4.5 casual (best human 2hr)
- 1,487: Opus 4.5 (11hr) **← Beat this to impress Anthropic**
- 1,363: Opus 4.5 improved harness

## Architecture Constraints
- VLEN = 8 (vector width)
- SLOT_LIMITS: ALU(12), VALU(6), LOAD(2), STORE(2), FLOW(1)
- Effects apply at END of cycle (enables parallel reads)
- Scratch space: 1536 words

## Key Files
- `perf_takehome.py`: KernelBuilder.build_kernel() - main optimization target
- `problem.py`: Machine simulator (read-only reference)
- `notes.md`: Optimization log and ideas

## Optimization Strategy
1. Multi-vector pipelining (use all 6 VALU slots)
2. Software pipelining (overlap load/compute/store)
3. Deep instruction scheduling

## Git Workflow
- Create branch per optimization phase
- Commit with cycle count in message
- Revert if regression, analyze, retry

## Common Patterns
### Pack 3 vectors in VALU (uses all 6 slots)
```python
body.append({
    "valu": [
        (op, dest_0, src_0, const),
        (op, dest_1, src_1, const),
        (op, dest_2, src_2, const),
    ]
})
```

### Overlap engines (VALU + LOAD in same cycle)
```python
body.append({
    "valu": [("+", v_addr, v_forest_p, v_idx)],
    "load": [("vload", v_next_idx, addr)],  # prefetch
})
```
