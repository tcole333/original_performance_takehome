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

### Optimization 4: Cross-Engine VLIW Packing
- **Cycles:** 13,902 (10.63x speedup over baseline, 1.04x over opt 3)
- Pack flow (vselect for wrap) with alu (store address computation)
- These use different engines and have no data dependencies
- Saves 1 cycle per iteration × 512 iterations = 512 cycles

---

## Ideas to Try Next

### High Priority
1. **Software Pipelining** - Prefetch next iteration's idx/val during hash phase
   - Hash phase is ~12 cycles with LOAD unit idle
   - Could load next iteration's data during this time
   - Requires double-buffering registers (v_idx/v_idx_next)
   - Attempted but incomplete - need to actually USE prefetched values

2. **Loop Unrolling** - Process 2+ chunks of 8 per inner loop iteration

### Medium Priority
3. **Fused Operations** - Use multiply_add where applicable in hash
4. **Pack more across engines** - Look for other flow/alu/valu overlaps

### Lower Priority / Speculative
5. **Better Gather Pattern** - Sort batch by tree index (complex, indices diverge each round)
6. **Reduce hash stages** - Algorithmic change, may not preserve semantics

### Ideas Evaluated and Rejected
- **Shuffle/repack for contiguous gathers** - No native sort primitives, indices diverge after each hash, overhead would exceed benefit

---

## Optimization Backlog (Track Across Sessions)

### High Priority (Implement Next)
- [ ] 3-vector VALU parallelism - use all 6 VALU slots during hash
- [ ] Software pipelining - overlap load/compute/store
- [ ] Store/Load overlap - 3-stage pipeline

### Medium Priority
- [ ] Hash stage unrolling
- [ ] Fused multiply-add in hash
- [ ] Address pre-computation during hash

### Low Priority / Speculative
- [ ] Runtime check for sequential indices (use vload)
- [ ] Loop interchange (process multiple rounds together)
- [ ] Memory layout restructuring

### Evaluated & Rejected
- Shuffle/sort for gather coalescing (no native support, indices diverge)
- Reduce hash stages (changes algorithm semantics)

---

## Experiments Log

### Session 1 (Initial optimizations)
- VLIW packing: 147,734 → 102,678 cycles
- SIMD vectorization: → 17,485 cycles
- Better VLIW packing: → 14,414 cycles
- Cross-engine packing: → 13,902 cycles

### Session 2 (Multi-vector pipelining)
- 3-vector VALU parallelism: → 7,822 cycles (18.9x over baseline)
- vselect→arithmetic: → 7,182 cycles (20.6x) - replace vselect with * and -
- Software pipelining: → 5,454 cycles (27.1x) - overlap Hash(N) with Gather(N+1)
- Bitwise AND: → 5,262 cycles (28.1x) - replace (val%2==0)?1:2 with 1+(val&1)
- Store||XOR: → 5,118 cycles (28.9x) - overlap store with XOR
- Triple engine overlap: → 4,991 cycles (29.6x) - Store||XOR||LoadAddrs
  - Bottleneck: Hash||Gather overlap (12 cycles), index update (5 cycles)
