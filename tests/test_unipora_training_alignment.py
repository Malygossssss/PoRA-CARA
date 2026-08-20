import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse(relative_path):
    path = PROJECT_ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} was not found")


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class UniPoRATrainingAlignmentTests(unittest.TestCase):
    def test_main_uses_unipora_independent_process_semantics(self):
        tree = _parse("main.py")
        call_names = {
            _call_name(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        attribute_names = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }

        self.assertNotIn("DistributedDataParallel", call_names)
        self.assertNotIn("no_sync", attribute_names)
        self.assertIn("ConstrainedMTLController", call_names)
        self.assertIn("mark_prompt_as_trainable", call_names)

    def test_mtl_build_does_not_partition_train_or_validation_data(self):
        tree = _parse("data/build.py")
        build_mtl = _find_function(tree, "build_mtl")
        call_names = {
            _call_name(node)
            for node in ast.walk(build_mtl)
            if isinstance(node, ast.Call)
        }

        self.assertNotIn("DistributedSampler", call_names)

        loader_calls = [
            node
            for node in ast.walk(build_mtl)
            if isinstance(node, ast.Call)
            and _call_name(node) in {"get_mtl_train_dataloader", "get_mtl_val_dataloader"}
        ]
        self.assertTrue(loader_calls)
        for call in loader_calls:
            self.assertFalse(any(keyword.arg == "sampler" for keyword in call.keywords))

    def test_mtl_dataloader_signatures_match_unipora(self):
        tree = _parse("data/mtl_ds.py")
        train_loader = _find_function(tree, "get_mtl_train_dataloader")
        val_loader = _find_function(tree, "get_mtl_val_dataloader")

        self.assertEqual([arg.arg for arg in train_loader.args.args], ["config", "dataset"])
        self.assertEqual([arg.arg for arg in val_loader.args.args], ["config", "dataset"])

        train_calls = [
            node for node in ast.walk(train_loader)
            if isinstance(node, ast.Call) and _call_name(node) == "DataLoader"
        ]
        self.assertEqual(len(train_calls), 1)
        shuffle = next(
            keyword.value
            for keyword in train_calls[0].keywords
            if keyword.arg == "shuffle"
        )
        self.assertIsInstance(shuffle, ast.Constant)
        self.assertIs(shuffle.value, True)

    def test_unipora_constrained_training_support_is_present(self):
        self.assertTrue((PROJECT_ROOT / "constrained_mtl.py").is_file())

        config_source = (PROJECT_ROOT / "config.py").read_text(encoding="utf-8")
        self.assertIn("_C.TRAIN.ENABLE_CONFLICT_RATIO", config_source)
        self.assertIn("_C.TRAIN.CONSTRAINED_MTL", config_source)


if __name__ == "__main__":
    unittest.main()
