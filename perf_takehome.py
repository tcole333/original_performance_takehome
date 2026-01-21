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

    def build_hash_simd_3wide(self, v_vals, v_tmp1s, v_tmp2s, round, batch_bases):
        """
        Build hash stages for 3 vectors simultaneously, using all 6 VALU slots.
        v_vals, v_tmp1s, v_tmp2s are lists of 3 vector register addresses.
        batch_bases is a list of 3 batch base indices.
        """
        bundles = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const1 = self.alloc_vector_const(val1)
            v_const3 = self.alloc_vector_const(val3)

            # VLIW Pack: All 6 tmp computations (2 ops × 3 vectors) - uses all 6 VALU slots
            bundles.append({
                "valu": [
                    (op1, v_tmp1s[0], v_vals[0], v_const1),
                    (op3, v_tmp2s[0], v_vals[0], v_const3),
                    (op1, v_tmp1s[1], v_vals[1], v_const1),
                    (op3, v_tmp2s[1], v_vals[1], v_const3),
                    (op1, v_tmp1s[2], v_vals[2], v_const1),
                    (op3, v_tmp2s[2], v_vals[2], v_const3),
                ]
            })
            # Final combine for all 3 vectors (3 VALU slots)
            bundles.append({
                "valu": [
                    (op2, v_vals[0], v_tmp1s[0], v_tmp2s[0]),
                    (op2, v_vals[1], v_tmp1s[1], v_tmp2s[1]),
                    (op2, v_vals[2], v_tmp1s[2], v_tmp2s[2]),
                ]
            })
            # Debug compares for all 3 vectors
            bundles.append({
                "debug": [
                    ("vcompare", v_vals[0], tuple((round, batch_bases[0] + j, "hash_stage", hi) for j in range(VLEN))),
                    ("vcompare", v_vals[1], tuple((round, batch_bases[1] + j, "hash_stage", hi) for j in range(VLEN))),
                    ("vcompare", v_vals[2], tuple((round, batch_bases[2] + j, "hash_stage", hi) for j in range(VLEN))),
                ]
            })

        return bundles

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        3-vector SIMD implementation processing 24 items per super-iteration.
        Uses all 6 VALU slots during hash for maximum throughput.
        Falls back to single-vector for remainder items.
        """
        # Scalar temps
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")

        # Additional scalar temps for 3-vector parallel address computation
        tmp_addr3 = self.alloc_scratch("tmp_addr3")
        tmp_addr4 = self.alloc_scratch("tmp_addr4")
        tmp_addr5 = self.alloc_scratch("tmp_addr5")
        tmp_addr6 = self.alloc_scratch("tmp_addr6")

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

        # 3 sets of vector registers for 3-vector parallel processing
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(3)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(3)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{i}", VLEN) for i in range(3)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(3)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(3)]
        v_tmp3 = [self.alloc_scratch(f"v_tmp3_{i}", VLEN) for i in range(3)]
        v_addr = [self.alloc_scratch(f"v_addr_{i}", VLEN) for i in range(3)]

        # Vector constants (broadcast from scalars)
        v_zero = self.alloc_vector_const(0)
        v_one = self.alloc_vector_const(1)
        v_two = self.alloc_vector_const(2)

        # Broadcast n_nodes to vector for comparison
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Broadcast forest_values_p once (hoisted out of loop)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting 3-vector SIMD loop"))

        body = []

        # Super-iteration stride: 3 vectors × VLEN = 24 items
        super_stride = VLEN * 3

        for round in range(rounds):
            # Process batch in super-iterations of 24 items
            batch_base = 0
            while batch_base + super_stride <= batch_size:
                # Process 3 vectors in parallel
                batch_bases = [batch_base, batch_base + VLEN, batch_base + 2 * VLEN]
                i_consts = [self.scratch_const(bb) for bb in batch_bases]

                # === Load addresses for all 3 vectors (6 ALU ops, 1 cycle) ===
                body.append({
                    "alu": [
                        ("+", tmp_addr, self.scratch["inp_indices_p"], i_consts[0]),
                        ("+", tmp_addr2, self.scratch["inp_values_p"], i_consts[0]),
                        ("+", tmp_addr3, self.scratch["inp_indices_p"], i_consts[1]),
                        ("+", tmp_addr4, self.scratch["inp_values_p"], i_consts[1]),
                        ("+", tmp_addr5, self.scratch["inp_indices_p"], i_consts[2]),
                        ("+", tmp_addr6, self.scratch["inp_values_p"], i_consts[2]),
                    ]
                })

                # === vload for all 3 vectors (6 vloads, 3 cycles at 2/cycle) ===
                body.append({
                    "load": [
                        ("vload", v_idx[0], tmp_addr),
                        ("vload", v_val[0], tmp_addr2),
                    ]
                })
                body.append({
                    "load": [
                        ("vload", v_idx[1], tmp_addr3),
                        ("vload", v_val[1], tmp_addr4),
                    ]
                })
                body.append({
                    "load": [
                        ("vload", v_idx[2], tmp_addr5),
                        ("vload", v_val[2], tmp_addr6),
                    ]
                })

                # Debug compares
                for vi in range(3):
                    body.append({
                        "debug": [
                            ("vcompare", v_idx[vi], tuple((round, batch_bases[vi] + j, "idx") for j in range(VLEN))),
                            ("vcompare", v_val[vi], tuple((round, batch_bases[vi] + j, "val") for j in range(VLEN))),
                        ]
                    })

                # === Gather addresses for all 3 vectors (3 VALU ops, 1 cycle) ===
                body.append({
                    "valu": [
                        ("+", v_addr[0], v_forest_p, v_idx[0]),
                        ("+", v_addr[1], v_forest_p, v_idx[1]),
                        ("+", v_addr[2], v_forest_p, v_idx[2]),
                    ]
                })

                # === Gather: 24 load_offsets at 2/cycle = 12 cycles ===
                for vi in range(3):
                    for offset in range(0, VLEN, 2):
                        body.append({
                            "load": [
                                ("load_offset", v_node_val[vi], v_addr[vi], offset),
                                ("load_offset", v_node_val[vi], v_addr[vi], offset + 1),
                            ]
                        })

                # Debug compares for node_val
                for vi in range(3):
                    body.append({
                        "debug": [
                            ("vcompare", v_node_val[vi], tuple((round, batch_bases[vi] + j, "node_val") for j in range(VLEN)))
                        ]
                    })

                # === XOR: val = val ^ node_val for all 3 vectors (3 VALU ops, 1 cycle) ===
                body.append({
                    "valu": [
                        ("^", v_val[0], v_val[0], v_node_val[0]),
                        ("^", v_val[1], v_val[1], v_node_val[1]),
                        ("^", v_val[2], v_val[2], v_node_val[2]),
                    ]
                })

                # === Hash using 3-wide SIMD (12 cycles, all 6 VALU slots) ===
                body.extend(self.build_hash_simd_3wide(v_val, v_tmp1, v_tmp2, round, batch_bases))

                # Debug compares for hashed_val
                for vi in range(3):
                    body.append({
                        "debug": [
                            ("vcompare", v_val[vi], tuple((round, batch_bases[vi] + j, "hashed_val") for j in range(VLEN)))
                        ]
                    })

                # === idx = 2*idx + (1 if val % 2 == 0 else 2) ===
                # % and * for all 3 vectors (6 VALU ops, 1 cycle)
                body.append({
                    "valu": [
                        ("%", v_tmp1[0], v_val[0], v_two),
                        ("*", v_idx[0], v_idx[0], v_two),
                        ("%", v_tmp1[1], v_val[1], v_two),
                        ("*", v_idx[1], v_idx[1], v_two),
                        ("%", v_tmp1[2], v_val[2], v_two),
                        ("*", v_idx[2], v_idx[2], v_two),
                    ]
                })

                # == for all 3 vectors (3 VALU ops, 1 cycle)
                body.append({
                    "valu": [
                        ("==", v_tmp1[0], v_tmp1[0], v_zero),
                        ("==", v_tmp1[1], v_tmp1[1], v_zero),
                        ("==", v_tmp1[2], v_tmp1[2], v_zero),
                    ]
                })

                # Replace vselect with arithmetic: select(cond, 1, 2) = 2 - cond
                # tmp3 = 2 - tmp1 for all 3 vectors (3 VALU ops, 1 cycle)
                body.append({
                    "valu": [
                        ("-", v_tmp3[0], v_two, v_tmp1[0]),
                        ("-", v_tmp3[1], v_two, v_tmp1[1]),
                        ("-", v_tmp3[2], v_two, v_tmp1[2]),
                    ]
                })

                # + for all 3 vectors (3 VALU ops, 1 cycle)
                body.append({
                    "valu": [
                        ("+", v_idx[0], v_idx[0], v_tmp3[0]),
                        ("+", v_idx[1], v_idx[1], v_tmp3[1]),
                        ("+", v_idx[2], v_idx[2], v_tmp3[2]),
                    ]
                })

                # Debug compares for next_idx
                for vi in range(3):
                    body.append({
                        "debug": [
                            ("vcompare", v_idx[vi], tuple((round, batch_bases[vi] + j, "next_idx") for j in range(VLEN)))
                        ]
                    })

                # === Wrap: idx = 0 if idx >= n_nodes else idx ===
                # < for all 3 vectors (3 VALU ops, 1 cycle)
                body.append({
                    "valu": [
                        ("<", v_tmp1[0], v_idx[0], v_n_nodes),
                        ("<", v_tmp1[1], v_idx[1], v_n_nodes),
                        ("<", v_tmp1[2], v_idx[2], v_n_nodes),
                    ]
                })

                # Replace vselect with arithmetic: select(cond, idx, 0) = idx * cond
                # Pack with address computation (6 VALU + 6 ALU in 1 cycle)
                body.append({
                    "valu": [
                        ("*", v_idx[0], v_idx[0], v_tmp1[0]),
                        ("*", v_idx[1], v_idx[1], v_tmp1[1]),
                        ("*", v_idx[2], v_idx[2], v_tmp1[2]),
                    ],
                    "alu": [
                        ("+", tmp_addr, self.scratch["inp_indices_p"], i_consts[0]),
                        ("+", tmp_addr2, self.scratch["inp_values_p"], i_consts[0]),
                        ("+", tmp_addr3, self.scratch["inp_indices_p"], i_consts[1]),
                        ("+", tmp_addr4, self.scratch["inp_values_p"], i_consts[1]),
                        ("+", tmp_addr5, self.scratch["inp_indices_p"], i_consts[2]),
                        ("+", tmp_addr6, self.scratch["inp_values_p"], i_consts[2]),
                    ]
                })

                # Debug compares for wrapped_idx
                for vi in range(3):
                    body.append({
                        "debug": [
                            ("vcompare", v_idx[vi], tuple((round, batch_bases[vi] + j, "wrapped_idx") for j in range(VLEN)))
                        ]
                    })

                # === vstore for all 3 vectors (6 vstores, 3 cycles at 2/cycle) ===
                body.append({
                    "store": [
                        ("vstore", tmp_addr, v_idx[0]),
                        ("vstore", tmp_addr2, v_val[0]),
                    ]
                })
                body.append({
                    "store": [
                        ("vstore", tmp_addr3, v_idx[1]),
                        ("vstore", tmp_addr4, v_val[1]),
                    ]
                })
                body.append({
                    "store": [
                        ("vstore", tmp_addr5, v_idx[2]),
                        ("vstore", tmp_addr6, v_val[2]),
                    ]
                })

                batch_base += super_stride

            # Handle remaining items with single-vector iterations
            while batch_base < batch_size:
                i_const = self.scratch_const(batch_base)

                # Use first set of vector registers for remainder
                body.append({
                    "alu": [
                        ("+", tmp_addr, self.scratch["inp_indices_p"], i_const),
                        ("+", tmp_addr2, self.scratch["inp_values_p"], i_const),
                    ]
                })
                body.append({
                    "load": [
                        ("vload", v_idx[0], tmp_addr),
                        ("vload", v_val[0], tmp_addr2),
                    ]
                })
                body.append({
                    "debug": [
                        ("vcompare", v_idx[0], tuple((round, batch_base + j, "idx") for j in range(VLEN))),
                        ("vcompare", v_val[0], tuple((round, batch_base + j, "val") for j in range(VLEN))),
                    ]
                })

                body.append({"valu": [("+", v_addr[0], v_forest_p, v_idx[0])]})

                for offset in range(0, VLEN, 2):
                    body.append({
                        "load": [
                            ("load_offset", v_node_val[0], v_addr[0], offset),
                            ("load_offset", v_node_val[0], v_addr[0], offset + 1),
                        ]
                    })
                body.append({
                    "debug": [
                        ("vcompare", v_node_val[0], tuple((round, batch_base + j, "node_val") for j in range(VLEN)))
                    ]
                })

                body.append({"valu": [("^", v_val[0], v_val[0], v_node_val[0])]})
                body.extend(self.build_hash_simd(v_val[0], v_tmp1[0], v_tmp2[0], round, batch_base))
                body.append({
                    "debug": [
                        ("vcompare", v_val[0], tuple((round, batch_base + j, "hashed_val") for j in range(VLEN)))
                    ]
                })

                body.append({
                    "valu": [
                        ("%", v_tmp1[0], v_val[0], v_two),
                        ("*", v_idx[0], v_idx[0], v_two),
                    ]
                })
                body.append({"valu": [("==", v_tmp1[0], v_tmp1[0], v_zero)]})
                # Replace vselect with arithmetic: select(cond, 1, 2) = 2 - cond
                body.append({"valu": [("-", v_tmp3[0], v_two, v_tmp1[0])]})
                body.append({"valu": [("+", v_idx[0], v_idx[0], v_tmp3[0])]})
                body.append({
                    "debug": [
                        ("vcompare", v_idx[0], tuple((round, batch_base + j, "next_idx") for j in range(VLEN)))
                    ]
                })

                body.append({"valu": [("<", v_tmp1[0], v_idx[0], v_n_nodes)]})
                # Replace vselect with arithmetic: select(cond, idx, 0) = idx * cond
                body.append({
                    "valu": [("*", v_idx[0], v_idx[0], v_tmp1[0])],
                    "alu": [
                        ("+", tmp_addr, self.scratch["inp_indices_p"], i_const),
                        ("+", tmp_addr2, self.scratch["inp_values_p"], i_const),
                    ]
                })
                body.append({
                    "debug": [
                        ("vcompare", v_idx[0], tuple((round, batch_base + j, "wrapped_idx") for j in range(VLEN)))
                    ]
                })
                body.append({
                    "store": [
                        ("vstore", tmp_addr, v_idx[0]),
                        ("vstore", tmp_addr2, v_val[0]),
                    ]
                })

                batch_base += VLEN

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
