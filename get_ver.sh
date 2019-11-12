#! /bin/bash

set -e
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
pushd 1>/dev/null $CURDIR
git describe
popd 1>/dev/null
