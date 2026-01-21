## Performance Optimization Log

### Baseline
- **Cycles:** 147,734
- Each instruction in its own cycle, no parallelism exploited

### Optimization 1: VLIW Packing (Scalar)
- **Cycles:** 102,678 (1.44x speedup)
- Pack independent ALU ops into single cycles (address calculations)
- Pack parallel LOADs (idx + val) and STOREs into single cycles
- Pack hash stage intermediate computations (2 ALU ops -> 1 cycle)
- Added `tmp_addr2` scratch register to avoid write conflicts

---

## Ideas to Try Next

### High Priority (Big Gains Expected)
1. **SIMD Vectorization** - Process VLEN=8 batch items per iteration instead of 1
   - Use vload/vstore for contiguous batch data
   - Use valu for parallel hash computations
   - Challenge: forest lookups are non-contiguous (gather pattern)

2. **Loop Unrolling** - Reduce loop overhead, expose more ILP

### Medium Priority
3. **Software Pipelining** - Overlap iterations (start next load while computing current)
4. **Batch Reordering** - Group items by tree index for better memory access patterns

### Lower Priority / Speculative
5. **Precompute Constants** - Move repeated scratch_const lookups out of loop
6. **Memory Layout Optimization** - Restructure data for better cache behavior
