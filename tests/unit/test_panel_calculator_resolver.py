import unittest

from macrosynergy.panel.panel_calculator import CalcList, SingleCalc


class TestPanelCalculatorResolver(unittest.TestCase):
    def test_single_calc_dependencies(self):
        calc = SingleCalc("A = B + C")
        self.assertEqual(calc.creates(), "A")
        self.assertIn("B", calc.dependencies())
        self.assertIn("C", calc.dependencies())

    def test_calc_list_simple_chain(self):
        calcs = [
            "A = B + 1",
            "B = C * 2",
            "C = D - 3",
        ]
        cl = CalcList(calcs, already_existing_vars=["D"])
        sorted_names = [c.creates() for c in cl.calcs]
        self.assertEqual(sorted_names, ["C", "B", "A"])

    def test_calc_list_multiple_independent_subgraphs(self):
        calcs = [
            "A = B + 1",
            "B = C * 2",
            "X = Y + 5",
            "Y = Z * 3",
        ]
        cl = CalcList(calcs, already_existing_vars=["C", "Z"])
        subgraphs = cl.get_independent_subgraphs()
        # Should be two subgraphs
        self.assertEqual(len(subgraphs), 2)
        # Each subgraph should be internally connected
        all_created = set([c.creates() for sg in subgraphs for c in sg])
        self.assertIn("A", all_created)
        self.assertIn("X", all_created)

    def test_calc_list_cycle_detection(self):
        calcs = [
            "A = B + 1",
            "B = C * 2",
            "C = A - 3",  # cycle: A -> B -> C -> A
        ]
        # Use an initial variable that makes the cycle reachable
        with self.assertRaises(ValueError):
            CalcList(calcs, already_existing_vars=["D"])

        # check that it resolves even if one of the variables is already existing
        calcs = [
            "A = B + 1",
            "B = C * 2",
            "C = A - 3",
        ]
        for exist_var in ["A", "B", "C"]:
            # Use an initial variable that makes the cycle reachable
            try:
                CalcList(calcs, already_existing_vars=[exist_var])
            except ValueError:
                self.fail(
                    f"Cycle detected with already existing variable {exist_var} should not raise an error"
                )

    def test_calc_list_parallel_blocks(self):
        calcs = [
            "A = B + 1",
            "B = C * 2",
            "C = D - 3",
        ]
        cl = CalcList(calcs, already_existing_vars=["D"])
        subgraphs = cl.get_independent_subgraphs()
        # Only one subgraph
        blocks = cl.get_subgraph_parallel_blocks(subgraphs[0])
        # Should be three layers, each with one calc
        self.assertEqual(len(blocks), 3)
        self.assertEqual([len(b) for b in blocks], [1, 1, 1])
        self.assertEqual([b[0].creates() for b in blocks], ["C", "B", "A"])

    def test_calc_list_feasibility(self):
        calcs = [
            "A = B + 1",
            "B = C * 2",
            "C = D - 3",
            "E = F + 1",  # F is missing
        ]
        cl = CalcList(calcs, already_existing_vars=["D"])
        created = [c.creates() for c in cl.feasible_calcs]
        self.assertIn("A", created)
        self.assertNotIn("E", created)

    def test_various_cases(self):
        # Generate 50+ cases with different dependency structures
        for i in range(20):
            # Simple chain
            calcs = [f"A{i} = B{i} + 1", f"B{i} = C{i} * 2", f"C{i} = D{i} - 3"]
            cl = CalcList(calcs, already_existing_vars=[f"D{i}"])
            sorted_names = [c.creates() for c in cl.calcs]
            self.assertEqual(sorted_names, [f"C{i}", f"B{i}", f"A{i}"])
        for i in range(10):
            # Parallel chains
            calcs = [
                f"A{i} = B{i} + 1",
                f"B{i} = C{i} * 2",
                f"X{i} = Y{i} + 5",
                f"Y{i} = Z{i} * 3",
            ]
            cl = CalcList(calcs, already_existing_vars=[f"C{i}", f"Z{i}"])
            subgraphs = cl.get_independent_subgraphs()
            self.assertEqual(len(subgraphs), 2)
            all_created = set([c.creates() for sg in subgraphs for c in sg])
            self.assertIn(f"A{i}", all_created)
            self.assertIn(f"X{i}", all_created)
        for i in range(10):
            # Star dependency
            calcs = [
                f"A{i} = X{i} + Y{i}",
                f"B{i} = X{i} + Y{i}",
                f"C{i} = X{i} + Y{i}",
            ]
            cl = CalcList(calcs, already_existing_vars=[f"X{i}", f"Y{i}"])
            sorted_names = [c.creates() for c in cl.calcs]
            self.assertEqual(set(sorted_names), {f"A{i}", f"B{i}", f"C{i}"})
        for i in range(10):
            # Diamond dependency
            calcs = [
                f"A{i} = B{i} + C{i}",
                f"B{i} = D{i} + 1",
                f"C{i} = D{i} + 2",
            ]
            cl = CalcList(calcs, already_existing_vars=[f"D{i}"])
            sorted_names = [c.creates() for c in cl.calcs]
            self.assertEqual(set(sorted_names), {f"B{i}", f"C{i}", f"A{i}"})
        for i in range(5):
            # Cycle detection
            calcs = [
                f"A{i} = B{i} + 1",
                f"B{i} = C{i} * 2",
                f"C{i} = A{i} - 3",
            ]
            with self.assertRaises(ValueError):
                CalcList(calcs, already_existing_vars=[f"D{i}"])

    def test_complex_operations(self):
        # Multiple operands and mixed dependencies
        calcs = [
            "A = B + C + D",
            "B = E * F - G",
            "C = H / I + J",
            "D = K * L * M",
            "E = 2",
            "F = 3",
            "G = 1",
            "H = 10",
            "I = 2",
            "J = 5",
            "K = 2",
            "L = 2",
            "M = 2",
        ]
        cl = CalcList(calcs, already_existing_vars=[])
        sorted_names = [c.creates() for c in cl.calcs]
        # All variables should be created in a valid topological order
        self.assertIn("A", sorted_names)
        self.assertIn("B", sorted_names)
        self.assertIn("C", sorted_names)
        self.assertIn("D", sorted_names)
        # Check that all leaf nodes are present
        for v in ["E", "F", "G", "H", "I", "J", "K", "L", "M"]:
            self.assertIn(v, sorted_names)

        # More complex: chain with multiple dependencies per step
        calcs2 = [
            "X = A + B + C",
            "Y = X + D + E",
            "Z = Y + F + G",
            "A = 1",
            "B = 2",
            "C = 3",
            "D = 4",
            "E = 5",
            "F = 6",
            "G = 7",
        ]
        cl2 = CalcList(calcs2, already_existing_vars=[])
        sorted_names2 = [c.creates() for c in cl2.calcs]
        self.assertIn("Z", sorted_names2)
        self.assertIn("Y", sorted_names2)
        self.assertIn("X", sorted_names2)
        # All leaf nodes
        for v in ["A", "B", "C", "D", "E", "F", "G"]:
            self.assertIn(v, sorted_names2)

        # Deeply nested dependencies
        calcs3 = [
            "A = B + C",
            "B = D + E",
            "C = F + G",
            "D = H + I",
            "E = J + K",
            "F = L + M",
            "G = N + O",
            "H = 1",
            "I = 2",
            "J = 3",
            "K = 4",
            "L = 5",
            "M = 6",
            "N = 7",
            "O = 8",
        ]
        cl3 = CalcList(calcs3, already_existing_vars=[])
        sorted_names3 = [c.creates() for c in cl3.calcs]
        self.assertIn("A", sorted_names3)
        for v in ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]:
            self.assertIn(v, sorted_names3)

        # Test with mixed operators and parentheses
        calcs4 = [
            "A = (B + C) * (D - E) / F",
            "B = 1",
            "C = 2",
            "D = 3",
            "E = 4",
            "F = 5",
        ]
        cl4 = CalcList(calcs4, already_existing_vars=[])
        sorted_names4 = [c.creates() for c in cl4.calcs]
        self.assertIn("A", sorted_names4)
        for v in ["B", "C", "D", "E", "F"]:
            self.assertIn(v, sorted_names4)

    def test_more_complex_operations(self):
        # Multiple levels of dependencies, mixed operators, and parentheses
        for i in range(10):
            calcs = [
                f"A{i} = (B{i} + C{i}) * (D{i} - E{i}) / F{i}",
                f"B{i} = G{i} * H{i} - I{i}",
                f"C{i} = J{i} / K{i} + L{i}",
                f"D{i} = M{i} * N{i} * O{i}",
                f"E{i} = P{i} - Q{i}",
                f"F{i} = R{i} + S{i}",
                f"G{i} = 2",
                f"H{i} = 3",
                f"I{i} = 1",
                f"J{i} = 10",
                f"K{i} = 2",
                f"L{i} = 5",
                f"M{i} = 2",
                f"N{i} = 2",
                f"O{i} = 2",
                f"P{i} = 10",
                f"Q{i} = 3",
                f"R{i} = 4",
                f"S{i} = 6",
            ]
            cl = CalcList(calcs, already_existing_vars=[])
            sorted_names = [c.creates() for c in cl.calcs]
            # All variables should be created in a valid topological order
            self.assertIn(f"A{i}", sorted_names)
            self.assertIn(f"B{i}", sorted_names)
            self.assertIn(f"C{i}", sorted_names)
            self.assertIn(f"D{i}", sorted_names)
            self.assertIn(f"E{i}", sorted_names)
            self.assertIn(f"F{i}", sorted_names)
            # Check that all leaf nodes are present
            for v in [
                f"G{i}",
                f"H{i}",
                f"I{i}",
                f"J{i}",
                f"K{i}",
                f"L{i}",
                f"M{i}",
                f"N{i}",
                f"O{i}",
                f"P{i}",
                f"Q{i}",
                f"R{i}",
                f"S{i}",
            ]:
                self.assertIn(v, sorted_names)

        # Test with deeply nested parentheses and mixed operations
        for i in range(5):
            calcs = [
                f"A{i} = (((B{i} + C{i}) * (D{i} - E{i})) / (F{i} + 1)) + ((G{i} - H{i}) * (I{i} + J{i}))",
                f"B{i} = 1",
                f"C{i} = 2",
                f"D{i} = 3",
                f"E{i} = 4",
                f"F{i} = 5",
                f"G{i} = 6",
                f"H{i} = 7",
                f"I{i} = 8",
                f"J{i} = 9",
            ]
            cl = CalcList(calcs, already_existing_vars=[])
            sorted_names = [c.creates() for c in cl.calcs]
            self.assertIn(f"A{i}", sorted_names)
            for v in [
                f"B{i}",
                f"C{i}",
                f"D{i}",
                f"E{i}",
                f"F{i}",
                f"G{i}",
                f"H{i}",
                f"I{i}",
                f"J{i}",
            ]:
                self.assertIn(v, sorted_names)

        # Test with a mix of chains, stars, and diamonds in one batch
        for i in range(5):
            calcs = [
                f"A{i} = B{i} + C{i}",
                f"B{i} = D{i} + E{i}",
                f"C{i} = F{i} + G{i}",
                f"D{i} = H{i} + I{i}",
                f"E{i} = J{i} + K{i}",
                f"F{i} = L{i} + M{i}",
                f"G{i} = N{i} + O{i}",
                f"H{i} = 1",
                f"I{i} = 2",
                f"J{i} = 3",
                f"K{i} = 4",
                f"L{i} = 5",
                f"M{i} = 6",
                f"N{i} = 7",
                f"O{i} = 8",
                f"P{i} = Q{i} + R{i}",  # star
                f"Q{i} = 9",
                f"R{i} = 10",
                f"S{i} = P{i} + A{i}",  # diamond
            ]
            cl = CalcList(calcs, already_existing_vars=[])
            sorted_names = [c.creates() for c in cl.calcs]
            for v in [
                f"A{i}",
                f"B{i}",
                f"C{i}",
                f"D{i}",
                f"E{i}",
                f"F{i}",
                f"G{i}",
                f"H{i}",
                f"I{i}",
                f"J{i}",
                f"K{i}",
                f"L{i}",
                f"M{i}",
                f"N{i}",
                f"O{i}",
                f"P{i}",
                f"Q{i}",
                f"R{i}",
                f"S{i}",
            ]:
                self.assertIn(v, sorted_names)

        # Test with a large number of chained dependencies
        for i in range(2):
            calcs = [f"A{i}_0 = 1"]
            for j in range(1, 30):
                calcs.append(f"A{i}_{j} = A{i}_{j-1} + 1")
            cl = CalcList(calcs, already_existing_vars=[])
            sorted_names = [c.creates() for c in cl.calcs]
            for j in range(30):
                self.assertIn(f"A{i}_{j}", sorted_names)


class TestPanelCalculatorResolverLarge(unittest.TestCase):
    def test_large_calculation_chain(self):
        all_calcs = []
        for i in range(10):
            calcs = [
                f"A{i} = (B{i} + C{i}) * (D{i} - E{i}) / F{i}",
                f"B{i} = G{i} * H{i} - I{i}",
                f"C{i} = J{i} / K{i} + L{i}",
                f"D{i} = M{i} * N{i} * O{i}",
                f"E{i} = P{i} - Q{i}",
                f"F{i} = R{i} + S{i}",
                f"G{i} = 2",
                f"H{i} = 3",
                f"I{i} = 1",
                f"J{i} = 10",
                f"K{i} = 2",
                f"L{i} = 5",
                f"M{i} = 2",
                f"N{i} = 2",
                f"O{i} = 2",
                f"P{i} = 10",
                f"Q{i} = 3",
                f"R{i} = 4",
                f"S{i} = 6",
            ]
            all_calcs.extend(calcs)

        cl = CalcList(all_calcs, already_existing_vars=[])
        sorted_names = [c.creates() for c in cl.calcs]
        # Check that all expected variables are present
        letters = [chr(c) for c in range(ord("A"), ord("S") + 1)]
        all_expected_vars = [f"{letter}{i}" for letter in letters for i in range(10)]
        self.assertEqual(set(sorted_names), set(all_expected_vars))
        # Check the independent subgraphs
        subgraphs = cl.get_independent_subgraphs()
        self.assertEqual(len(subgraphs), 10)
        # Each subgraph should contain the correct variables
        for sg in subgraphs:
            sg_vars = set(c.creates() for c in sg)
            # Extract the index from any variable name in the subgraph
            any_var = next(iter(sg_vars))
            idx = "".join(filter(str.isdigit, any_var))
            expected_vars = {f"{letter}{idx}" for letter in letters}
            self.assertEqual(sg_vars, expected_vars)


if __name__ == "__main__":
    unittest.main()
