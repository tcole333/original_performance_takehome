"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def build_vliw(self, bundles: list[dict[str, list[tuple]]]):
        """
        Build instruction bundles for VLIW execution.
        Each bundle is a dict mapping engine names to lists of slots.
        """
        return bundles

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_hash_vliw(self, val_hash_addr, tmp1, tmp2, round, i):
        """
        Build hash stages with VLIW packing.
        Each hash stage: tmp1 = op1(val, const1), tmp2 = op3(val, const3), val = op2(tmp1, tmp2)
        The first two ops are independent and can be packed.
        """
        bundles = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # VLIW Pack: Both tmp computations are independent (1 cycle)
            bundles.append({
                "alu": [
                    (op1, tmp1, val_hash_addr, self.scratch_const(val1)),
                    (op3, tmp2, val_hash_addr, self.scratch_const(val3)),
                ]
            })
            # Final combine depends on both (1 cycle)
            bundles.append({"alu": [(op2, val_hash_addr, tmp1, tmp2)]})
            # Debug compare must be separate
            bundles.append({"debug": [("compare", val_hash_addr, (round, i, "hash_stage", hi))]})

        return bundles

    def alloc_vector_const(self, val, name=None):
        """Allocate a vector constant by broadcasting a scalar value."""
        scalar_addr = self.scratch_const(val, name)
        if (val, "vec") not in self.const_map:
            vec_addr = self.alloc_scratch(f"v_{name}" if name else None, VLEN)
            self.add("valu", ("vbroadcast", vec_addr, scalar_addr))
            self.const_map[(val, "vec")] = vec_addr
        return self.const_map[(val, "vec")]

    def build_hash_simd(self, v_val, v_tmp1, v_tmp2, round, batch_base):
        """
        Build hash stages using SIMD valu operations.
        Processes VLEN values in parallel.
        """
        bundles = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const1 = self.alloc_vector_const(val1)
            v_const3 = self.alloc_vector_const(val3)

            # VLIW Pack: Both tmp computations are independent (1 cycle)
            bundles.append({
                "valu": [
                    (op1, v_tmp1, v_val, v_const1),
                    (op3, v_tmp2, v_val, v_const3),
                ]
            })
            # Final combine depends on both (1 cycle)
            bundles.append({"valu": [(op2, v_val, v_tmp1, v_tmp2)]})
            # Debug compare for all VLEN values
            bundles.append({
                "debug": [
                    ("vcompare", v_val, tuple((round, batch_base + j, "hash_stage", hi) for j in range(VLEN)))
                ]
            })

        return bundles

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        SIMD implementation processing VLEN items per iteration.
        Uses vload/vstore for contiguous data, scalar loads for gather.
        """
        # Scalar temps
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Scratch space addresses for memory layout
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector registers (VLEN=8 slots each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)  # For gather addresses

        # Vector constants (broadcast from scalars)
        v_zero = self.alloc_vector_const(0)
        v_one = self.alloc_vector_const(1)
        v_two = self.alloc_vector_const(2)

        # Broadcast n_nodes to vector for comparison
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting SIMD loop"))

        body = []

        for round in range(rounds):
            # Process batch in chunks of VLEN=8
            for batch_base in range(0, batch_size, VLEN):
                i_const = self.scratch_const(batch_base)

                # === Load idx[batch_base:batch_base+8] with vload ===
                # Compute base address: inp_indices_p + batch_base
                body.append({"alu": [("+", tmp_addr, self.scratch["inp_indices_p"], i_const)]})
                body.append({"load": [("vload", v_idx, tmp_addr)]})
                body.append({
                    "debug": [
                        ("vcompare", v_idx, tuple((round, batch_base + j, "idx") for j in range(VLEN)))
                    ]
                })

                # === Load val[batch_base:batch_base+8] with vload ===
                body.append({"alu": [("+", tmp_addr, self.scratch["inp_values_p"], i_const)]})
                body.append({"load": [("vload", v_val, tmp_addr)]})
                body.append({
                    "debug": [
                        ("vcompare", v_val, tuple((round, batch_base + j, "val") for j in range(VLEN)))
                    ]
                })

                # === Gather: compute addresses for forest lookup ===
                # v_addr[j] = forest_values_p + v_idx[j]
                # Need to broadcast forest_values_p to vector first
                v_forest_p = self.alloc_scratch("v_forest_p", VLEN) if "v_forest_p" not in self.scratch else self.scratch["v_forest_p"]
                if "v_forest_p" not in self.scratch:
                    self.scratch["v_forest_p"] = v_forest_p
                body.append({"valu": [("vbroadcast", v_forest_p, self.scratch["forest_values_p"])]})
                body.append({"valu": [("+", v_addr, v_forest_p, v_idx)]})

                # === 8 scalar loads for gather (2 per cycle = 4 cycles) ===
                for offset in range(0, VLEN, 2):
                    body.append({
                        "load": [
                            ("load_offset", v_node_val, v_addr, offset),
                            ("load_offset", v_node_val, v_addr, offset + 1),
                        ]
                    })
                body.append({
                    "debug": [
                        ("vcompare", v_node_val, tuple((round, batch_base + j, "node_val") for j in range(VLEN)))
                    ]
                })

                # === val = myhash(val ^ node_val) using valu ===
                body.append({"valu": [("^", v_val, v_val, v_node_val)]})
                body.extend(self.build_hash_simd(v_val, v_tmp1, v_tmp2, round, batch_base))
                body.append({
                    "debug": [
                        ("vcompare", v_val, tuple((round, batch_base + j, "hashed_val") for j in range(VLEN)))
                    ]
                })

                # === idx = 2*idx + (1 if val % 2 == 0 else 2) ===
                # v_tmp1 = v_val % 2
                body.append({"valu": [("%", v_tmp1, v_val, v_two)]})
                # v_tmp1 = (v_tmp1 == 0)
                body.append({"valu": [("==", v_tmp1, v_tmp1, v_zero)]})
                # v_tmp3 = select(v_tmp1, 1, 2)
                body.append({"flow": [("vselect", v_tmp3, v_tmp1, v_one, v_two)]})
                # v_idx = v_idx * 2
                body.append({"valu": [("*", v_idx, v_idx, v_two)]})
                # v_idx = v_idx + v_tmp3
                body.append({"valu": [("+", v_idx, v_idx, v_tmp3)]})
                body.append({
                    "debug": [
                        ("vcompare", v_idx, tuple((round, batch_base + j, "next_idx") for j in range(VLEN)))
                    ]
                })

                # === idx = 0 if idx >= n_nodes else idx ===
                body.append({"valu": [("<", v_tmp1, v_idx, v_n_nodes)]})
                body.append({"flow": [("vselect", v_idx, v_tmp1, v_idx, v_zero)]})
                body.append({
                    "debug": [
                        ("vcompare", v_idx, tuple((round, batch_base + j, "wrapped_idx") for j in range(VLEN)))
                    ]
                })

                # === Store results with vstore ===
                body.append({"alu": [("+", tmp_addr, self.scratch["inp_indices_p"], i_const)]})
                body.append({"store": [("vstore", tmp_addr, v_idx)]})

                body.append({"alu": [("+", tmp_addr, self.scratch["inp_values_p"], i_const)]})
                body.append({"store": [("vstore", tmp_addr, v_val)]})

        self.instrs.extend(body)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
