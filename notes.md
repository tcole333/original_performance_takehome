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

### Optimization 2: SIMD Vectorization
- **Cycles:** 17,485 (8.45x speedup over baseline, 5.87x over opt 1)
- Process VLEN=8 batch items per iteration (32 iterations instead of 256)
- vload/vstore for contiguous idx[] and val[] arrays (1 cycle for 8 values)
- 8 scalar loads for forest gather (4 cycles, 2 loads/cycle)
- valu for all hash computations (same cycles, 8x throughput)
- vselect for conditional operations

### Optimization 3: Better VLIW Packing in SIMD
- **Cycles:** 14,414 (10.25x speedup over baseline, 1.21x over opt 2)
- Pack idx/val vloads into same cycle (4 → 2 cycles)
- Hoist v_forest_p broadcast out of loop (saves 1 cycle × 512 iterations)
- Pack % and * operations together (2 → 1 cycle)
- Pack vstores into same cycle (4 → 2 cycles)

---

## Ideas to Try Next

### High Priority
1. **Loop Unrolling** - Process 2+ chunks of 8 per inner loop iteration
2. **Software Pipelining** - Overlap iterations (start next vload while computing current)

### Medium Priority
3. **Fused Operations** - Use multiply_add where applicable in hash
4. **More ALU/VALU Packing** - Look for other independent operations

### Lower Priority / Speculative
5. **Better Gather Pattern** - Sort batch by tree index (complex, likely not worth it)
6. **Precompute batch_base constants** - Minor optimization
