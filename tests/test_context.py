from unittest import TestCase
from contextual_encoders import TreeContext


class TestTreeContext(TestCase):
    def test_create_one_layer_tree_context(self):
        tree_context = TreeContext("Gender")
        tree_context.add_concept("Male")
        tree_context.add_concept("Female")

        tree = tree_context.get_tree()

        self.assertEqual(tree_context.get_root(), "Gender", "Should be Gender")
        self.assertTrue("Male" in tree.nodes, "Should contain Male")
        self.assertTrue("Female" in tree.nodes, "Should contain Female")

    def test_create_two_layer_tree_context(self):
        tree_context = TreeContext("Color")
        tree_context.add_concept("Dark")
        tree_context.add_concept("Light")
        tree_context.add_concept("Yellow", "Light", weight=0.5)
        tree_context.add_concept("Darkblue", "Dark", weight=0.4)

        tree = tree_context.get_tree()

        self.assertEqual(tree_context.get_root(), "Color", "Should be Color")
        self.assertTrue("Yellow" in tree.nodes, "Should contain Dark")
        self.assertTrue("Darkblue" in tree.nodes, "Should contain Female")
        self.assertTrue(
            tree.has_edge("Light", "Yellow"),
            "Should have edge between Light and Yellow",
        )
        self.assertTrue(
            tree.has_edge("Dark", "Darkblue"),
            "Should have edge between Dark and Darkblue",
        )
        self.assertEqual(
            tree["Light"]["Yellow"]["weight"], 0.5, "Edge should have weight 0.5"
        )
        self.assertEqual(
            tree["Dark"]["Darkblue"]["weight"], 0.4, "Edge should have weight 0.4"
        )
