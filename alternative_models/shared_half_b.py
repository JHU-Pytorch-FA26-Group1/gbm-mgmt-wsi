"""Inspect Tian's shared Half B folder; --copy downloads and verifies its package.

Uses an existing rclone OneDrive login. No credentials are printed or copied into
receipts. Drive/root overrides apply only to each command, not remote settings.
Only reads remote files; never submits jobs. Standard library only.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from . import half_b_package as pre

SHARE_URL = (
    "https://livejohnshopkins-my.sharepoint.com/:f:/r/personal/tliu130_jh_edu/"
    "Documents/GBM_packages/B_qc-v5?d=w089cb1943680481d992ee7fab26b8bce"
    "&csf=1&web=1&e=qOJgs4"
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the OAuth Authorization header to a redirect target.
        return None


def graph_item(token, url=SHARE_URL):
    parsed = urllib.parse.urlsplit(url)
    pre.require(parsed.scheme == "https" and parsed.netloc == "livejohnshopkins-my.sharepoint.com"
                and parsed.path == "/:f:/r/personal/tliu130_jh_edu/Documents/GBM_packages/B_qc-v5",
                "Expected Tian's exact Hopkins Half B folder URL")
    encoded = "u!" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    request = urllib.request.Request(
        "https://graph.microsoft.com/v1.0/shares/" + encoded + "/driveItem"
        "?$select=id,name,folder,parentReference,remoteItem",
        headers={"Authorization": "Bearer " + token, "Accept": "application/json",
                 "Prefer": "redeemSharingLinkIfNecessary"})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
            raw = response.read(2*1024*1024 + 1)
        pre.require(len(raw) <= 2*1024*1024, "Unexpectedly large Graph response")
        item = json.loads(raw)
    except urllib.error.HTTPError as error:
        raise ValueError(
            f"Microsoft Graph returned HTTP {error.code}. No package was copied. "
            "Check that this folder opens under your Hopkins account; this error "
            "does not mean the folder is empty. Do not paste tokens here.") from None
    except (urllib.error.URLError, json.JSONDecodeError):
        raise ValueError("Could not read shared-folder metadata from Microsoft Graph") from None
    if "remoteItem" in item:
        item = item["remoteItem"]
    pre.require(item.get("name") == "B_qc-v5" and "folder" in item,
                "The link did not resolve to a B_qc-v5 folder")
    drive = item.get("parentReference", {}).get("driveId", "")
    root = item.get("id", "")
    pre.require(all(isinstance(i, str) and re.fullmatch(r"[A-Za-z0-9!_.~-]+", i)
                    for i in (drive, root)), "Missing/unexpected drive or folder ID")
    return {"drive_id": drive, "root_folder_id": root, "folder_name": item["name"]}


class Rclone:
    def __init__(self, executable, config, remote):
        pre.require(re.fullmatch(r"[A-Za-z0-9_-]+", remote) is not None,
                    "Remote must be a configured name, without a colon or path")
        self.prefix = [str(executable), "--config", str(config)]
        self.remote = remote

    def run(self, args, folder=None, visible=False):
        command = self.prefix + list(args)
        if folder:
            command += ["--onedrive-drive-id", folder["drive_id"],
                        "--onedrive-root-folder-id", folder["root_folder_id"],
                        "--onedrive-drive-type", "business"]
        result = subprocess.run(command, check=False,
                                stdout=None if visible else subprocess.PIPE,
                                stderr=None if visible else subprocess.PIPE)
        pre.require(result.returncode == 0,
                    f"rclone {args[0]} failed (exit {result.returncode}). "
                    "Check the existing remote's sign-in and folder access.")
        return result.stdout

    def token(self):
        # An ordinary metadata request lets rclone refresh its stored login token.
        self.run(["about", self.remote + ":", "--json"])
        try:
            config = json.loads(self.run(["config", "dump"]))
            section = config[self.remote]
            pre.require(section.get("type") == "onedrive", "Expected a OneDrive remote")
            pre.require(not section.get("tenant_url"), "Expected a Microsoft Graph OneDrive login")
            token = section["token"]
            token = json.loads(token) if isinstance(token, str) else token
            value = token["access_token"]
            pre.require(isinstance(value, str) and value, "OneDrive login has no access token")
            return value
        except (KeyError, TypeError, json.JSONDecodeError):
            raise ValueError("Could not read the existing rclone OneDrive login") from None


def validate_manifest(raw, a_package, repo):
    pre.require(len(raw) <= 2*1024*1024, "Unexpectedly large package manifest")
    manifest = json.loads(raw)
    pre.require(manifest["kind"] == "package" and manifest["subset"] == "B"
                and manifest["complete"] is True, "Expected a complete portable Half B package")
    pre.require(manifest["tiling_version"] == "tiling-v7"
                and set(manifest["policy_version"].split("+")) == {"qc-v5", "qc-v5-cptac"},
                "Expected tiling-v7 and both qc-v5 policies")
    original = json.loads((a_package / "package_manifest.json").read_text())
    pre.require(original["subset"] == "A" and original["complete"] is True,
                "Expected a complete Half A comparison package")
    for key in pre.SHARED:
        pre.require(manifest[key] == original[key], f"A/B package setting differs: {key}")
    pre.require(pre.sha256(repo / "manifests/patient_folds.csv") == manifest["folds_sha256"],
                "Repository fold map differs")
    pre.require(pre.sha256(repo / "configs/tiling.yaml") == manifest["config_sha256"],
                "Repository tiling config differs")
    files = manifest["files"]
    pre.require(isinstance(files, dict) and files, "Empty package file inventory")
    for name, record in files.items():
        pre.safe_name(name)
        pre.require(name == PurePosixPath(name).as_posix() and name != "package_manifest.json"
                    and not any(ord(c) < 32 or ord(c) == 127 for c in name), "Unsafe manifest filename")
        pre.require(type(record["bytes"]) is int and record["bytes"] >= 0
                    and isinstance(record["sha256"], str)
                    and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "Invalid manifest size/hash")
    pre.require({"modeling_patches.csv", "patient_index.csv", "slide_status.csv", "export_settings.json"}
                <= files.keys(), "Package metadata incomplete")
    return manifest


class LocalPackage:
    """Small adapter reusing the existing ownership/population validator on disk."""
    def __init__(self, path):
        self.path = path

    def open(self, name):
        return (self.path / name).open("rb")


def verify_stage(stage, raw, manifest, repo):
    pre.require((stage / "package_manifest.json").read_bytes() == raw,
                "Downloaded manifest differs from the inspected copy")
    for i, (name, record) in enumerate(manifest["files"].items(), 1):
        path = stage / name
        pre.require(path.is_file() and not path.is_symlink()
                    and path.stat().st_size == record["bytes"]
                    and pre.sha256(path) == record["sha256"], f"Package content/hash mismatch: {name}")
        if i % 20 == 0 or i == len(manifest["files"]):
            print(f"SHA-256 verified {i}/{len(manifest['files'])} package files", flush=True)
    return pre.population(LocalPackage(stage), "", manifest, repo)


def copy_package(rclone, folder, raw, manifest, repo, base):
    packages = base / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    out, stage = packages / "B_qc-v5", packages / ".B_qc-v5.download"
    record = base / "experiments/alternative-models/half-b-onedrive"
    fingerprint = dict(folder, manifest_sha256=hashlib.sha256(raw).hexdigest(), remote=rclone.remote)
    pre.require(not out.exists() and not out.is_symlink(), "B_qc-v5 already exists; refusing to overwrite it")
    pre.require(not stage.is_symlink(), "Staging path must not be a symlink")
    marker = record / "source.json"
    if stage.exists():
        pre.require(marker.is_file() and json.loads(marker.read_text()) == fingerprint,
                    "Staging folder belongs to a different/unrecorded source")
        pre.require(all(not p.is_symlink() for p in stage.rglob("*")), "Symlink found in staging folder")
    else:
        pre.require(not marker.exists(), "An earlier download record exists without its staging folder")
        record.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(fingerprint, indent=2) + "\n")
        stage.mkdir(mode=0o700)
    needed = sum(max(0, r["bytes"] - ((stage / n).stat().st_size if (stage / n).is_file() else 0))
                 for n, r in manifest["files"].items())
    # rclone replaces individual files using temporary files; allow the largest
    # two files as additional headroom for the two parallel transfers on a retry.
    largest = sum(sorted((r["bytes"] for r in manifest["files"].values()), reverse=True)[:2])
    pre.require(shutil.disk_usage(packages).free >= needed + largest + 512*1024**2,
                "Insufficient disk space for download plus temporary-file headroom")
    files = record / "files.txt"
    files.write_text("package_manifest.json\n" + "".join(n + "\n" for n in manifest["files"]))
    (record / "inspected-package_manifest.json").write_bytes(raw)
    # Traverse the known folder and use listed object IDs; direct file lookups
    # under an overridden shared root fail on this Run:ai/rclone combination.
    rclone.run(["copy", rclone.remote + ":", str(stage), "--files-from-raw", str(files), "--no-traverse=false",
                "--checksum", "--transfers", "2", "--checkers", "4", "--progress", "--stats", "30s",
                "--log-file", str(record / "copy.log"), "--log-level", "INFO"], folder, visible=True)
    population = verify_stage(stage, raw, manifest, repo)
    pre.require(not out.exists() and not out.is_symlink(), "Destination appeared during download")
    stage.rename(out)
    receipt = dict(fingerprint, complete=True, package=str(out),
                   files_sha256_verified=len(manifest["files"]), population=population,
                   code_sha256={str(p): pre.sha256(p) for p in (Path(__file__), Path(pre.__file__))},
                   completed_at=datetime.now(timezone.utc).isoformat(),
                   scope="Manifest file sizes/SHA-256 and canonical B ownership/folds; no generation or training")
    (record / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/workspace/persist/crivas4-microscopy"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--copy", action="store_true", help="Download, verify and publish B_qc-v5; otherwise inspect only")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        executable = shutil.which("rclone") or args.base / "bin/rclone"
        config = Path(os.environ.get("RCLONE_CONFIG", args.base / "rclone-config/rclone.conf"))
        client = Rclone(executable, config, "jhu-onedrive")
        folder = graph_item(client.token())
        entries = json.loads(client.run(["lsjson", client.remote + ":", "--max-depth", "1"], folder))
        metadata = [e for e in entries if e.get("Path") == "package_manifest.json" and not e.get("IsDir")]
        pre.require(len(metadata) == 1 and 0 < metadata[0]["Size"] <= 2*1024*1024,
                    "The shared folder has no usable root-level package_manifest.json")
        # Keep the filesystem rooted at the folder. Carlos verified that the
        # direct file form fails with "found file when looking for folder".
        raw = client.run(["cat", client.remote + ":", "--include", "/package_manifest.json",
                          "--max-depth", "1", "--count", "2097153"], folder)
        manifest = validate_manifest(raw, args.base / "packages/A_qc-v5", args.repo)
        print(json.dumps(dict(folder, package_counts=manifest["counts"],
                              manifest_files=len(manifest["files"]),
                              expanded_GiB=round(sum(r["bytes"] for r in manifest["files"].values())/2**30, 2),
                              compatible_with_half_a=True, remote_write_operations=0,
                              action="download and verify" if args.copy else "inspection only"), indent=2), flush=True)
        if args.copy:
            print(json.dumps(copy_package(client, folder, raw, manifest, args.repo, args.base), indent=2))
        else:
            print("Inspection passed. Add --copy to download; no package files were downloaded.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Half B OneDrive access stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
