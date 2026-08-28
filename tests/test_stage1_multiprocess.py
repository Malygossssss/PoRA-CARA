import ast
import unittest
from pathlib import Path
from unittest import mock

from ag_mtlora.stage1_multiprocess import (
    Stage1ProcessContext,
    build_multi_process_failure_manifest,
    build_multi_process_manifest,
    build_rank_failure_manifest,
    build_rank_artifact_manifest,
    get_failure_report_path,
    resolve_process_context,
    resolve_stage1_output_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _call_names(relative_path):
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


class Stage1ProcessContextTests(unittest.TestCase):
    def test_resolves_torchrun_environment_and_rank_seed(self):
        context = resolve_process_context(
            cli_local_rank=0,
            environ={"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"},
        )

        self.assertEqual(context.rank, 1)
        self.assertEqual(context.world_size, 2)
        self.assertEqual(context.local_rank, 1)
        self.assertTrue(context.is_multi_process)
        self.assertFalse(context.is_primary)
        self.assertEqual(context.effective_seed(100), 101)

    def test_single_process_defaults_are_backward_compatible(self):
        context = resolve_process_context(cli_local_rank=3, environ={})

        self.assertEqual(context, Stage1ProcessContext(rank=0, world_size=1, local_rank=3))
        self.assertFalse(context.is_multi_process)
        self.assertTrue(context.is_primary)

    def test_rejects_invalid_rank_or_world_size(self):
        with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
            resolve_process_context(environ={"WORLD_SIZE": "0"})
        with self.assertRaisesRegex(ValueError, "RANK"):
            resolve_process_context(environ={"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "0"})
        with self.assertRaisesRegex(ValueError, "LOCAL_RANK"):
            resolve_process_context(environ={"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "-1"})


class Stage1OutputPathTests(unittest.TestCase):
    def test_single_process_keeps_legacy_output_layout(self):
        config_output = str(PROJECT_ROOT / "output_for_test")
        context = Stage1ProcessContext(rank=0, world_size=1, local_rank=0)
        paths = resolve_stage1_output_paths(
            config_output=config_output,
            context=context,
            timestamp="20260820_120000",
        )

        expected_root = str(
            Path(config_output) / "ag_mtlora_stage1_prepare" / "run_20260820_120000"
        )
        self.assertEqual(paths.run_root, expected_root)
        self.assertEqual(paths.rank_output_root, expected_root)

    def test_multi_process_uses_rank_isolated_directories(self):
        context = Stage1ProcessContext(rank=1, world_size=2, local_rank=1)
        paths = resolve_stage1_output_paths(
            config_output=str(PROJECT_ROOT / "output_for_test"),
            context=context,
            timestamp="20260820_120000",
        )

        self.assertEqual(paths.rank_output_root, str(Path(paths.run_root) / "rank_1"))
        self.assertEqual(
            paths.rank_artifact_manifest_path,
            str(Path(paths.run_root) / "rank_1" / "stage1_artifacts.json"),
        )
        self.assertEqual(
            paths.root_manifest_path,
            str(Path(paths.run_root) / "stage1_multi_process_manifest.json"),
        )

    def test_multi_process_resume_requires_existing_rank_directory(self):
        run_root = PROJECT_ROOT / "existing_run_for_test"
        rank_root = run_root / "rank_1"
        context = Stage1ProcessContext(rank=1, world_size=2, local_rank=1)

        with mock.patch(
            "ag_mtlora.stage1_multiprocess.os.path.isdir",
            side_effect=lambda path: Path(path) == run_root,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "rank_1"):
                resolve_stage1_output_paths(
                    config_output="unused",
                    context=context,
                    resume_stage1_dir=str(run_root),
                )

        with mock.patch(
            "ag_mtlora.stage1_multiprocess.os.path.isdir",
            side_effect=lambda path: Path(path) in {run_root, rank_root},
        ):
            paths = resolve_stage1_output_paths(
                config_output="unused",
                context=context,
                resume_stage1_dir=str(run_root),
            )
        self.assertEqual(paths.rank_output_root, str(rank_root))


class Stage1ManifestTests(unittest.TestCase):
    def test_failure_manifest_points_to_rank_specific_report(self):
        context = Stage1ProcessContext(rank=1, world_size=2, local_rank=1)
        paths = resolve_stage1_output_paths(
            config_output=str(PROJECT_ROOT / "output_for_test"),
            context=context,
            timestamp="20260828_120000",
        )
        failure_report_path = get_failure_report_path(context, paths)
        manifest = build_rank_failure_manifest(
            context=context,
            paths=paths,
            effective_seed=43,
            failure_report_path=failure_report_path,
            error_type="Stage1NumericalError",
            error_message="non-finite total loss",
        )

        self.assertEqual(Path(failure_report_path).name, "failure_report_rank1.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["rank"], 1)
        self.assertEqual(manifest["effective_seed"], 43)
        self.assertEqual(manifest["failure_report_path"], failure_report_path)
        self.assertEqual(manifest["artifacts"], {})

    def test_primary_rank_can_build_root_failure_manifest(self):
        context = Stage1ProcessContext(rank=0, world_size=2, local_rank=0)
        paths = resolve_stage1_output_paths(
            config_output=str(PROJECT_ROOT / "output_for_test"),
            context=context,
            timestamp="20260828_120000",
        )
        failure_report_path = get_failure_report_path(context, paths)

        manifest = build_multi_process_failure_manifest(
            context=context,
            paths=paths,
            failure_report_path=failure_report_path,
            error_type="RuntimeError",
            error_message="boom",
        )

        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_rank"], 0)
        self.assertEqual(manifest["failure_report_path"], failure_report_path)

    def test_rank_zero_is_the_canonical_result(self):
        context = Stage1ProcessContext(rank=0, world_size=2, local_rank=0)
        paths = resolve_stage1_output_paths(
            config_output=str(PROJECT_ROOT / "output_for_test"),
            context=context,
            timestamp="20260820_120000",
        )
        artifacts = {
            "resolved_config_path": str(Path(paths.rank_output_root) / "resolved.yaml"),
            "post_affinity_checkpoint_path": str(
                Path(paths.rank_output_root) / "post_affinity_checkpoint.pth"
            ),
        }
        rank_manifest = build_rank_artifact_manifest(
            context=context,
            paths=paths,
            effective_seed=42,
            artifacts=artifacts,
        )
        root_manifest = build_multi_process_manifest(
            context=context,
            paths=paths,
            canonical_artifacts=artifacts,
        )

        self.assertEqual(rank_manifest["rank"], 0)
        self.assertEqual(rank_manifest["effective_seed"], 42)
        self.assertEqual(root_manifest["mode"], "unipora_independent_processes")
        self.assertEqual(root_manifest["canonical_rank"], 0)
        self.assertEqual(root_manifest["canonical_stage1_dir"], paths.rank_output_root)
        self.assertEqual(
            root_manifest["canonical_resolved_config_path"],
            artifacts["resolved_config_path"],
        )
        self.assertEqual(len(root_manifest["ranks"]), 2)

    def test_non_primary_rank_cannot_build_root_manifest(self):
        context = Stage1ProcessContext(rank=1, world_size=2, local_rank=1)
        paths = resolve_stage1_output_paths(
            config_output=str(PROJECT_ROOT / "output_for_test"),
            context=context,
            timestamp="20260820_120000",
        )
        with self.assertRaisesRegex(ValueError, "rank 0"):
            build_multi_process_manifest(context, paths, {})


class Stage1LauncherContractTests(unittest.TestCase):
    def test_launcher_initializes_process_group_without_ddp_or_data_sharding(self):
        launcher_calls = _call_names("scripts/ag_mtlora_stage1_prepare.py")
        stage1_calls = _call_names("ag_mtlora/stage1.py")
        launcher_source = (PROJECT_ROOT / "scripts/ag_mtlora_stage1_prepare.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("init_process_group", launcher_calls)
        self.assertIn("set_device", launcher_calls)
        self.assertIn("broadcast_object_list", launcher_calls)
        self.assertIn("destroy_process_group", launcher_calls)
        self.assertNotIn("DistributedDataParallel", launcher_calls | stage1_calls)
        self.assertNotIn("DistributedSampler", launcher_calls | stage1_calls)
        self.assertNotIn("dist.barrier", launcher_source)

    def test_rank_seed_is_written_back_to_stage1_runtime_config(self):
        source = (PROJECT_ROOT / "scripts/ag_mtlora_stage1_prepare.py").read_text(encoding="utf-8")

        self.assertIn("config.SEED = effective_seed", source)

    def test_launcher_accepts_both_local_rank_spellings(self):
        source = (PROJECT_ROOT / "scripts/ag_mtlora_stage1_prepare.py").read_text(encoding="utf-8")
        self.assertIn('"--local_rank"', source)
        self.assertIn('"--local-rank"', source)

    def test_launcher_persists_failures_and_emits_terminal_markers(self):
        launcher_calls = _call_names("scripts/ag_mtlora_stage1_prepare.py")
        source = (PROJECT_ROOT / "scripts/ag_mtlora_stage1_prepare.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("write_failure_report", launcher_calls)
        self.assertIn("build_rank_failure_manifest", launcher_calls)
        self.assertIn("logger.exception", source)
        self.assertIn("STAGE1_ABORTED", source)
        self.assertIn("STAGE1_COMPLETED", source)
        self.assertIn("raise\n", source)


if __name__ == "__main__":
    unittest.main()
