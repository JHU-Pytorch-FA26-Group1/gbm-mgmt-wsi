"""Mocked authenticated access and local transfer fixtures; no cloud credentials."""
import io
import json
from pathlib import Path
import subprocess
import unittest
import urllib.error
from unittest.mock import patch

from alternative_models import shared_half_b as shared
from alternative_models.tests import test_half_b_package as fixtures


class SharedB(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HalfB()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.repo = fixtures.REPO
        self.raw = json.dumps(self.fixture.manifest).encode()
        self.manifest = shared.validate_manifest(self.raw, self.fixture.original, self.repo)
        self.folder = dict(drive_id="b!fixture-drive", root_folder_id="fixture-folder", folder_name="B_qc-v5")

    def local_client(self, corrupt=False, interrupt=False):
        fixture = self.fixture
        raw = self.raw
        class Client:
            remote = "jhu-onedrive"
            calls = []
            def run(inner, args, folder=None, visible=False):
                inner.calls.append((args, folder))
                if interrupt:
                    raise ValueError("simulated interruption")
                destination = Path(args[2])
                (destination / "package_manifest.json").write_bytes(raw)
                for name, data in fixture.payloads.items():
                    path = destination / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x"*len(data) if corrupt and name.endswith(".h5") else data)
        return Client()

    def test_copy_scoped_manifest_only_and_verify_before_publish(self):
        client = self.local_client()
        receipt = shared.copy_package(client, self.folder, self.raw, self.manifest, self.repo, self.root)
        self.assertTrue(receipt["complete"])
        self.assertTrue((self.root / "packages/B_qc-v5/tiles/test.h5").exists())
        self.assertFalse((self.root / "packages/.B_qc-v5.download").exists())
        args, folder = client.calls[0]
        self.assertEqual(folder, self.folder)
        self.assertEqual(args[0], "copy")
        self.assertIn("--no-traverse=false", args)
        names = set(Path(args[args.index("--files-from-raw")+1]).read_text().splitlines())
        self.assertEqual(names, set(self.manifest["files"]) | {"package_manifest.json"})
        with self.assertRaisesRegex(ValueError, "already exists"):
            shared.copy_package(client, self.folder, self.raw, self.manifest, self.repo, self.root)

    def test_same_size_corruption_never_publishes(self):
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            shared.copy_package(self.local_client(corrupt=True), self.folder, self.raw,
                                self.manifest, self.repo, self.root)
        self.assertFalse((self.root / "packages/B_qc-v5").exists())
        self.assertTrue((self.root / "packages/.B_qc-v5.download").exists())

    def test_inspection_reads_manifest_through_filtered_folder(self):
        calls = []

        def run(client, args, folder=None, visible=False):
            calls.append(args)
            if args[0] == "lsjson":
                return json.dumps([dict(Path="package_manifest.json", IsDir=False,
                                        Size=len(self.raw))]).encode()
            if args[0] == "cat":
                self.assertEqual(args[1], "jhu-onedrive:")
                self.assertEqual(args[args.index("--include")+1], "/package_manifest.json")
                self.assertEqual(args[args.index("--max-depth")+1], "1")
                self.assertEqual(args[args.index("--count")+1], "2097153")
                return self.raw
            raise AssertionError("Inspection must not copy files")

        with patch.object(shared.sys, "argv", ["shared_half_b", "--base", str(self.root), "--repo", str(self.repo)]), \
             patch.object(shared.Rclone, "token", return_value="fixture-secret"), \
             patch.object(shared, "graph_item", return_value=self.folder), \
             patch.object(shared.Rclone, "run", run), \
             patch.object(shared, "validate_manifest", return_value=self.manifest), \
             patch.object(shared.sys, "stdout", io.StringIO()) as output:
            self.assertEqual(shared.main(), 0)
        self.assertEqual([c[0] for c in calls], ["lsjson", "cat"])
        self.assertIn("Inspection passed", output.getvalue())
        self.assertNotIn("fixture-secret", output.getvalue())

    def test_interruption_can_resume_and_different_source_is_refused(self):
        with self.assertRaisesRegex(ValueError, "interruption"):
            shared.copy_package(self.local_client(interrupt=True), self.folder, self.raw,
                                self.manifest, self.repo, self.root)
        other = dict(self.folder, root_folder_id="different")
        with self.assertRaisesRegex(ValueError, "different/unrecorded"):
            shared.copy_package(self.local_client(), other, self.raw, self.manifest, self.repo, self.root)
        shared.copy_package(self.local_client(), self.folder, self.raw, self.manifest, self.repo, self.root)
        self.assertTrue((self.root / "packages/B_qc-v5").exists())

    def test_unsafe_manifest_and_incompatible_policy_are_refused(self):
        for name in ("../escape", "bad\nname", "/absolute", "double//separator"):
            m = json.loads(self.raw)
            m["files"][name] = dict(bytes=1, sha256="a"*64)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                shared.validate_manifest(json.dumps(m).encode(), self.fixture.original, self.repo)
        m = json.loads(self.raw)
        m["policy_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "differs"):
            shared.validate_manifest(json.dumps(m).encode(), self.fixture.original, self.repo)

    def test_graph_metadata_only_no_auth_redirects_or_token_echo(self):
        item = dict(id="folder", name="B_qc-v5", folder={}, parentReference=dict(driveId="b!drive"))
        opener = unittest.mock.Mock()
        opener.open.return_value = io.BytesIO(json.dumps(item).encode())
        with patch.object(shared.urllib.request, "build_opener", return_value=opener) as build:
            result = shared.graph_item("fixture-secret")
        self.assertEqual(result["root_folder_id"], "folder")
        self.assertIsInstance(build.call_args.args[0], shared.NoRedirect)
        request = opener.open.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://graph.microsoft.com/v1.0/shares/u!"))
        self.assertEqual(request.get_header("Authorization"), "Bearer fixture-secret")
        self.assertNotIn("fixture-secret", json.dumps(result))
        self.assertIsNone(shared.NoRedirect().redirect_request(request, None, 302, "", {}, "https://other.example"))
        error = urllib.error.HTTPError(request.full_url, 403, "fixture-secret", {}, io.BytesIO())
        opener.open.side_effect = error
        with patch.object(shared.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "HTTP 403") as caught:
                shared.graph_item("fixture-secret")
        self.assertNotIn("fixture-secret", str(caught.exception))
        with self.assertRaisesRegex(ValueError, "exact Hopkins"):
            shared.graph_item("fixture-secret", "https://other.example/folder")

    def test_rclone_flags_are_per_command_and_config_dump_is_captured(self):
        client = shared.Rclone("rclone", self.root / "private.conf", "jhu-onedrive")
        config = {"jhu-onedrive": dict(type="onedrive", token=json.dumps(dict(access_token="fixture-secret")))}
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(config).encode(), stderr=b"")
        with patch.object(shared.subprocess, "run", return_value=result) as run:
            self.assertEqual(client.token(), "fixture-secret")
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["stdout"], subprocess.PIPE)
                self.assertNotIn("fixture-secret", str(call.args))
            client.run(["lsjson", "jhu-onedrive:"], self.folder)
            command = run.call_args.args[0]
            self.assertIn("--onedrive-root-folder-id", command)
            self.assertIn("--onedrive-drive-id", command)
            self.assertNotIn("update", command)


if __name__ == "__main__":
    unittest.main()
