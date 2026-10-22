#!/bin/bash
# mfem3_launch.sh <compiler> <args...>: CMake compiler launcher for the ejwork/mfem_builds plugin builds.
# - A source listed in $MFEM3_FALLBACK (one repo-relative path per line) is compiled without the Reactant plugin
#   (plain clang CUDA, as in the native build); logged as FALLBACK.
# - Any other failed compile is retried once after a pause (high -j builds hit load-dependent failures); logged as RETRY.
# Log: $MFEM3_LAUNCH_LOG (default logs/mfem3_launch.log).
LOG=${MFEM3_LAUNCH_LOG:-/home/wmoses/ejwork/logs/mfem3_launch.log}
src=""; for a in "$@"; do case "$a" in *.cpp|*.cu|*.cc) src=$a ;; esac; done
rel=${src#/home/wmoses/git/mfem/}; rel=${rel#/mnt3/wmoses/git/mfem/}
if [ -n "$src" ] && [ -f "$MFEM3_FALLBACK" ] && grep -qx "$rel" "$MFEM3_FALLBACK"; then
  args=(); skip=0; prev=""
  for a in "$@"; do
    if [ $skip -gt 0 ]; then skip=$((skip-1)); continue; fi
    case "$a" in
      -fplugin=*) continue ;;
      -mllvm) prev=-mllvm; continue ;;
    esac
    if [ "$prev" = -mllvm ]; then
      prev=""
      case "$a" in
        -raising-plugin-path|-reactant-backend|-reactant-device-lib) skip=2; continue ;;  # and the following "-mllvm <value>"
        -reactant-*) continue ;;
        *) args+=(-mllvm "$a"); continue ;;
      esac
    fi
    args+=("$a")
  done
  echo "$(date -Is) FALLBACK $rel" >> $LOG
  exec "${args[@]}"
fi
"$@" && exit 0
rc=$?
echo "$(date -Is) RETRY rc=$rc $rel" >> $LOG
sleep $(( 5 + RANDOM % 20 ))
"$@"; rc2=$?
[ $rc2 -ne 0 ] && echo "$(date -Is) FAILED-TWICE rc=$rc2 $rel" >> $LOG
exit $rc2
