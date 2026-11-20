#!/usr/bin/env bash
# Copy a package from OneDrive or Google Drive to local disk with rclone.
#
# Usage:
#   scripts/fetch_package.sh --login
#   scripts/fetch_package.sh --login --src "gdrive:643-deep-learning-project/packages/B_qc-v5"
#   scripts/fetch_package.sh --src "jh-onedrive:GBM_packages/B_qc-v5" [--dry-run]
#
# Options:
#   --src REMOTE:PATH  package folder on the rclone remote
#   --dest DIR         local folder (default: data/packages/<last part of --src>); rerun to resume
#   --config FILE      rclone config file (default: data/rclone.conf, holds your sign-in token)
#   --login            sign in through the browser (adds the remote from scripts/rclone.template.conf
#                      if it is missing; use again when the sign-in expires)
#   --transfers N      parallel transfers (default: rclone's, 4)
#   --dry-run          list what would be copied, and stop
#
# Steps: size and free-space check, then rclone copy --checksum.
# A --dest or --config inside the repo is refused unless git ignores it; data/ is ignored.
# Exit codes: 0 done, 1 copy failed, 2 setup or usage problem. Logs: logs/fetch_*.log.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${GBM_PACKAGE_SRC:-}"
dest="${GBM_PACKAGE_DEST:-}"
transfers=""
dry_run=0
login=0

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        --config) config_arg="$2"; shift 2 ;;
        --transfers) transfers="$2"; shift 2 ;;
        --login) login=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v rclone >/dev/null || { echo "rclone is not installed" >&2; exit 2; }
template="$repo/scripts/rclone.template.conf"
export RCLONE_CONFIG="${config_arg:-${RCLONE_CONFIG:-$repo/data/rclone.conf}}"
config_file="$RCLONE_CONFIG"
case "$config_file" in
    "$repo"/*)
        if ! git -C "$repo" check-ignore -q -- "$config_file"; then
            echo "the rclone config holds a sign-in token and git does not ignore it: $config_file" >&2
            exit 2
        fi ;;
esac

if [[ $login -eq 1 ]]; then
    remote="jh-onedrive"
    [[ "$src" == *:* ]] && remote="${src%%:*}"
    add_template=0
    if ! rclone listremotes 2>/dev/null | grep -qx "${remote}:"; then
        if [[ "$remote" == "jh-onedrive" && -f "$template" ]]; then
            add_template=1
            cmd=(rclone config reconnect "${remote}:")
        elif [[ "$remote" == "gdrive" ]]; then
            cmd=(rclone config create "$remote" drive scope=drive.readonly)
        else
            cmd=(rclone config create "$remote" onedrive)
        fi
    else
        cmd=(rclone config reconnect "${remote}:")
    fi
    [[ $add_template -eq 1 ]] && echo "==> add the remote from $template to $config_file"
    echo "==> login: ${cmd[*]}   (config: $config_file)"
    [[ $dry_run -eq 1 ]] && exit 0
    umask 077
    mkdir -p "$(dirname "$config_file")"
    [[ $add_template -eq 1 ]] && grep -v '^#' "$template" >> "$config_file"
    "${cmd[@]}"
    exit 0
fi
if [[ -z "$src" ]]; then
    echo "--src is required" >&2
    exit 2
fi
if [[ "$src" != *:* ]]; then
    echo "--src must look like REMOTE:PATH: $src" >&2
    exit 2
fi
[[ -z "$dest" ]] && dest="$repo/data/packages/${src##*/}"
if [[ -n "$transfers" && ! "$transfers" =~ ^[1-9][0-9]*$ ]]; then
    echo "--transfers must be a positive integer: $transfers" >&2
    exit 2
fi
mkdir -p "$dest" "$repo/logs"
dest_abs="$(cd "$dest" && pwd)"
case "$dest_abs/" in
    "$repo"/*)
        if ! git -C "$repo" check-ignore -q -- "$dest_abs/modeling_patches.csv"; then
            echo "--dest is inside the repo and .gitignore does not ignore it: $dest_abs" >&2
            exit 2
        fi ;;
esac
name="$(basename "$dest")"
log="$repo/logs/fetch_${name}_$(date +%Y%m%d_%H%M%S).log"

echo "==> size: rclone size \"$src\""
size_json="$(rclone size "$src" --json)" || { echo "==> cannot read the source; check the remote and path, or sign in with --login (config: $config_file)" >&2; exit 2; }
needed="$(sed -n 's/.*"bytes": *\([0-9][0-9]*\).*/\1/p' <<< "$size_json")"
[[ -n "$needed" ]] || { echo "==> could not read the source size from: $size_json" >&2; exit 2; }
have_local="$(du -sk "$dest" | cut -f1)"
free_kb="$(df -Pk "$dest" | awk 'NR==2 {print $4}')"
remaining_kb=$(( needed / 1024 - have_local ))
echo "    source: $needed bytes; already in $dest: ${have_local} KB; free on that disk: ${free_kb} KB"
if [[ $remaining_kb -gt $free_kb ]]; then
    echo "==> not enough free space: need about ${remaining_kb} KB more, ${free_kb} KB free" >&2
    exit 2
fi

copy=(rclone copy "$src" "$dest" --checksum --progress --log-file "$log")
[[ -n "$transfers" ]] && copy+=(--transfers "$transfers")
[[ $dry_run -eq 1 ]] && copy+=(--dry-run)
echo "==> copy: ${copy[*]}"
set +e
"${copy[@]}"
code=$?
set -e
if [[ $code -ne 0 ]]; then
    echo "==> copy stopped with exit code $code; see $log. Run the same command again to resume." >&2
    exit 1
fi
if [[ $dry_run -eq 1 ]]; then
    echo "==> dry run done; nothing copied"
else
    echo "==> done: $dest"
fi
