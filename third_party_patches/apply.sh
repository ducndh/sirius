#!/usr/bin/env bash
# Applies the cuCascade patch this branch needs (a direct I/O option for the disk tier, and a
# wake-up of reservation waiters when a fixed-size host block is returned). Run it after
# `git submodule update --init --recursive`; running it again is a no-op.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
patch=$root/third_party_patches/cucascade_direct_io_and_wake.patch
sub=$root/cucascade
want=$(git -C "$root" rev-parse HEAD:cucascade)
have=$(git -C "$sub" rev-parse HEAD 2>/dev/null || echo none)
if [ "$have" != "$want" ]; then
  echo "cucascade is at $have, the patch is for $want: run git submodule update --init --recursive first" >&2
  exit 1
fi
if git -C "$sub" apply -R --check "$patch" 2>/dev/null; then
  echo "cucascade patch already applied"
  exit 0
fi
git -C "$sub" apply "$patch"
git -C "$sub" apply -R --check "$patch"
echo "cucascade patch applied"
