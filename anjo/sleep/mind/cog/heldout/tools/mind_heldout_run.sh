#!/bin/zsh
# MIND held-out runs (prereg §3), from the mind-freeze worktree. usage: mind_heldout_run.sh local|gemini
# A run is done when its eval exits 0 and validate.py accepts its records; a done run is skipped.
# Attempts are counted per run: a 2nd attempt (the one resume the prereg allows) needs MIND_RESUME_NOTE,
# which is registered in the prereg ledger first; a 3rd is refused, and the run is reported as failed.
set -u; setopt pipefail
STAGE=$1
FREEZE=3451998c4b8f56c9f0eb53a1ad827f1073d7fee1
S=/private/tmp/claude-501/-Users-k/f8dbbf52-df21-4aab-acb6-f342415fbeb3/scratchpad
R=$S/mind_research/runs; H=$S/mind_research/heldout; T=$S/mind_research/heldout_tools
FZ=/Users/k/dev/everthing/Anjo-mind-freeze; PY=/Users/k/dev/everthing/Anjo-sleep/.venv/bin/python
SL=/Users/k/dev/everthing/Anjo-freeze   # the prereg ledger's code
B=$R/snap_tally10k_e8/cog_core_final.pt; P2C=$R/snap_indep_final/cog_core_platt.pt
LOG=$H/run.log; mkdir -p $H/attempts
say() { echo "$(date +%H:%M:%S) $*" | tee -a $LOG; }
cd $FZ || exit 1
[[ $(git rev-parse HEAD) == $FREEZE && -z $(git status --porcelain) ]] || { say "REFUSED: not a clean checkout of $FREEZE"; exit 1; }
(cd $T && shasum -a 256 -c --quiet MANIFEST.sha256) || { say "REFUSED: a tool differs from MANIFEST.sha256"; exit 1; }
registered() {  # registered KIND FILE: FILE's sha256 is the latest ledger entry of KIND
  (cd $SL && ANJO_DATA_DIR=/Users/k/AnjoSleepData .venv/bin/python -c "
import sys
from anjo.sleep.prereg import ledger as L
es = [e for e in L.read_entries() if e.kind == '$1']
sys.exit(0 if es and es[-1].sha256 == L.sha256_file('$2') else 1)")
}
registered mind_tools $T/MANIFEST.sha256 || { say "REFUSED: MANIFEST.sha256 is not the registered mind_tools"; exit 1; }
registered mind_seeds $H/seeds.json || { say "REFUSED: seeds.json is not the registered mind_seeds"; exit 1; }
[[ $(shasum -a 256 $B | cut -c1-64) == 4009ebe1db7e8088c5e7cc67c98809900b648968777b4ee10481b1f327566a00 ]] || { say "REFUSED: B sha"; exit 1; }
[[ $(shasum -a 256 $P2C | cut -c1-64) == d06ff4ff2b95c83b7bc665d00b105b7f5a09b02d782aa2963627228b0cc2963f ]] || { say "REFUSED: P2 sha"; exit 1; }
seeds() { python3 -c "import json; print(','.join(map(str, json.load(open('$H/seeds.json'))['$1'])))"; }
MAIN=$(seeds main); MAINQ=$(seeds main_gemini); IND=$(seeds indep); INDQ=$(seeds indep_gemini)
export PYTHONPATH=$FZ ANJO_DATA_DIR=/Users/k/AnjoSleepData
valid() { $PY $T/validate.py $H/seeds.json $H run $1 > /dev/null 2>&1; }
attempt() {  # attempt LABEL: count this attempt, or refuse
  local f=$H/attempts/$1 n; n=$(cat $f 2>/dev/null || echo 0)
  if (( n >= 2 )); then say "REFUSED $1: two attempts already; report it as failed"; return 1; fi
  if (( n == 1 )); then
    [[ -n ${MIND_RESUME_NOTE:-} ]] || { say "REFUSED $1: a resume needs MIND_RESUME_NOTE (the infrastructure failure)"; return 1; }
    (cd $SL && ANJO_DATA_DIR=/Users/k/AnjoSleepData .venv/bin/python -c "
import hashlib, sys
from anjo.sleep.prereg import ledger as L
note = sys.argv[1] + ': ' + sys.argv[2]
L.register('mind_heldout_resume', hashlib.sha256(note.encode()).hexdigest(), note)" "$1" "$MIND_RESUME_NOTE") \
      || { say "REFUSED $1: could not register the resume"; return 1; }
    say "resume $1 registered: $MIND_RESUME_NOTE"
  fi
  echo $((n + 1)) > $f; ATTEMPT=$((n + 1))
}
ev() {  # ev LABEL -- eval args...
  local label=$1; shift 2
  if valid $label; then say "skip $label (done)"; return 0; fi
  attempt $label || return 1
  say "start $label (attempt $ATTEMPT)"
  $PY -m anjo.sleep.mind.cog eval "$@" > $H/$label.eval.$ATTEMPT.log 2>&1; local rc=$?
  (( rc == 0 )) && valid $label && { say "done $label"; return 0; }
  say "FAILED $label rc=$rc ($($PY $T/validate.py $H/seeds.json $H run $label 2>&1 | tail -1))"; return 1
}
local_runs() {
  local p
  for p in online_lr population counter surprise_lr; do
    ev m_$p -- --predictor $p --seeds $MAIN --out $H/m_$p --label m_$p --workers 4 || return 1; done
  for p in mind_nonight mind mind_nofast mind_k1; do
    ev m_$p -- --predictor $p --seeds $MAIN --out $H/m_$p --label m_$p --workers 4 --set ckpt=$B || return 1; done
  ev t3_mind_voi -- --predictor mind_nonight --inquirer mind_voi --seeds $MAIN --out $H/t3_mind_voi --workers 3 \
     --set ckpt=$B --set lam=1.0 || return 1
  for p in max_uncertainty random; do
    ev t3_$p -- --predictor mind_nonight --inquirer $p --seeds $MAIN --out $H/t3_$p --workers 3 --set ckpt=$B || return 1; done
  ev q_mind_nonight -- --predictor mind_nonight --seeds $MAINQ --query-every 7 --out $H/q_mind_nonight \
     --label q_mind_nonight --workers 4 --set ckpt=$B || return 1
  ev q_online_lr -- --predictor online_lr --seeds $MAINQ --query-every 7 --out $H/q_online_lr --label q_online_lr --workers 4 || return 1
  for p in online_lr population counter surprise_lr; do
    ev i_$p -- --world indep --predictor $p --seeds $IND --out $H/i_$p --label i_$p --workers 4 || return 1; done
  ev i_zeroshot -- --world indep --predictor mind_nonight --seeds $IND --out $H/i_zeroshot --label i_zeroshot \
     --workers 4 --set ckpt=$B || return 1
  ev i_p2 -- --world indep --predictor mind_nonight --seeds $IND --out $H/i_p2 --label i_p2 --workers 4 --set ckpt=$P2C || return 1
  ev iq_p2 -- --world indep --predictor mind_nonight --seeds $INDQ --query-every 7 --out $H/iq_p2 --label iq_p2 \
     --workers 4 --set ckpt=$P2C || return 1
  ev iq_online_lr -- --world indep --predictor online_lr --seeds $INDQ --query-every 7 --out $H/iq_online_lr \
     --label iq_online_lr --workers 4 || return 1
}
gem() {  # gem LABEL PREDICTOR SEEDS A B [extra eval args]: a paid run under a $6 hold, then its frozen Platt map
  local label=$1 pred=$2 sd=$3 a=$4 b=$5; shift 5
  local out=$H/$label
  if valid $label; then say "skip $label (done)"; else
    attempt $label || return 1
    local job="cog_gemini:heldout_$label:a$ATTEMPT"
    mkdir -p $out; say "start $label (paid, attempt $ATTEMPT)"
    $PY - <<PYH || { say "FAILED $label: hold refused"; return 1; }
from anjo.sleep.budget import default_ledger, Bound
r = default_ledger().reserve("$job", ("vertex_mind_usd",), Bound(rate_per_hour=6.0, timeout_s=3600.0))
if r is None: raise SystemExit("vertex_mind_usd refused the hold")
PYH
    COG_VERTEX_SPEND=$out/spend.$ATTEMPT.json COG_VERTEX_CAP_USD=6 $PY -m anjo.sleep.mind.cog eval --predictor $pred \
      --seeds $sd --query-every 7 --out $out --label $label --workers 6 "$@" \
      --set client=anjo.sleep.mind.cog.vertex_client:complete --set cache_dir=$H/gemini_cache > $H/$label.eval.$ATTEMPT.log 2>&1
    local rc=$?
    $PY - <<PYH
import json, pathlib
from anjo.sleep.budget import default_ledger, Bound
p = pathlib.Path("$out/spend.$ATTEMPT.json"); usd = json.loads(p.read_text())["usd"] if p.exists() else 0.0
default_ledger().reserve("$job", ("vertex_mind_usd",), Bound(rate_per_hour=6.0, timeout_s=3600.0)).settle(usd)
print("settled $job", round(usd, 4))
PYH
    (( rc == 0 )) && valid $label || { say "FAILED $label rc=$rc"; return 1; }
    say "done $label"
  fi
  [[ -f $out/${label}_platt.jsonl ]] || $PY $T/apply_frozen_platt.py $out/$label.jsonl $out/${label}_platt.jsonl \
    ${label}_platt $a $b | tee -a $LOG || { say "FAILED $label: Platt map"; return 1; }
}
gemini_runs() {
  gem q_gemini gemini $MAINQ 0.577884 0.375832 || return 1
  gem q_gemini_stats gemini_stats $MAINQ 0.598332 0.328086 || return 1
  gem iq_gemini gemini $INDQ 0.694137 -0.072288 --world indep || return 1
  gem iq_gemini_stats gemini_stats $INDQ 0.661167 -0.134386 --world indep || return 1
}
say "STAGE $STAGE start (freeze ${FREEZE:0:8})"
case $STAGE in
  local) local_runs; rc=$? ;;
  gemini) gemini_runs; rc=$? ;;
  *) echo "usage: $0 local|gemini"; exit 2 ;;
esac
say "STAGE $STAGE EXIT rc=$rc"; exit $rc
