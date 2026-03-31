#!/bin/sh
set -e
CCC=/opt/ccc/bin/ccc-x86
case "$(uname -m)" in
    aarch64) CCC=/opt/ccc/bin/ccc-arm ;;
    riscv64) CCC=/opt/ccc/bin/ccc-riscv ;;
    i686 | i386) CCC=/opt/ccc/bin/ccc-i686 ;;
esac
exec "$CCC" "$@"
