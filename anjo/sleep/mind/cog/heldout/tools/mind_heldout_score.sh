#!/bin/zsh
# MIND held-out scoring (prereg §4-5), once, from the mind-freeze worktree. Validates every record file,
# writes $H/score/*.json, then SCORES.json (the endpoints on the registered bars) atomically, last.
# SCORES.json is the score of the prereg; its sha256 gets registered. MIND_H is for dry runs only.
set -u; setopt pipefail
FREEZE=3451998c4b8f56c9f0eb53a1ad827f1073d7fee1
S=/private/tmp/claude-501/-Users-k/f8dbbf52-df21-4aab-acb6-f342415fbeb3/scratchpad
H=${MIND_H:-$S/mind_research/heldout}; T=$S/mind_research/heldout_tools; O=$H/score
FZ=/Users/k/dev/everthing/Anjo-mind-freeze; PY=/Users/k/dev/everthing/Anjo-sleep/.venv/bin/python
SL=/Users/k/dev/everthing/Anjo-freeze
cd $FZ || exit 1
[[ $(git rev-parse HEAD) == $FREEZE && -z $(git status --porcelain) ]] || { echo "REFUSED: not a clean checkout of $FREEZE"; exit 1; }
(cd $T && shasum -a 256 -c --quiet MANIFEST.sha256) || { echo "REFUSED: a tool differs from MANIFEST.sha256"; exit 1; }
if [[ -z ${MIND_H:-} ]]; then
  for kf in "mind_tools $T/MANIFEST.sha256" "mind_seeds $H/seeds.json"; do
    (cd $SL && ANJO_DATA_DIR=/Users/k/AnjoSleepData .venv/bin/python -c "
import sys
from anjo.sleep.prereg import ledger as L
kind, path = sys.argv[1], sys.argv[2]
es = [e for e in L.read_entries() if e.kind == kind]
sys.exit(0 if es and es[-1].sha256 == L.sha256_file(path) else 1)" ${=kf}) || { echo "REFUSED: $kf is not registered"; exit 1; }
  done
fi
[[ -e $O/SCORES.json ]] && { echo "REFUSED: already scored (scored once)"; exit 1; }
export PYTHONPATH=$FZ
$PY $T/validate.py $H/seeds.json $H all || exit 1
mkdir -p $O
rep() { local out=$1 ref=$2; shift 2; local r=(); [[ $ref != - ]] && r=(--reference $ref)
  $PY -m anjo.sleep.mind.cog report --records "$@" --out $O/$out.json --n-boot 10000 $r > $O/$out.log 2>&1 \
    || { echo "FAILED report $out"; exit 1; }; }
j() { echo $H/$1/$1.jsonl; }
rep P1_main m_online_lr $(j m_online_lr) $(j m_population) $(j m_counter) $(j m_surprise_lr) $(j m_mind_nonight)
rep P5_night m_mind_nonight $(j m_mind_nonight) $(j m_mind)
rep P6_lesions m_mind $(j m_mind) $(j m_mind_nofast) $(j m_mind_k1)
rep P4_t3 - $H/t3_mind_voi/mind_nonight.jsonl "$H/t3_mind_voi/mind_nonight+mind_voi.jsonl" \
  "$H/t3_max_uncertainty/mind_nonight+max_uncertainty.jsonl" "$H/t3_random/mind_nonight+random.jsonl"
rep P2_qe7 q_gemini_stats_platt $(j q_mind_nonight) $(j q_online_lr) $(j q_gemini) $(j q_gemini_stats) \
  $H/q_gemini/q_gemini_platt.jsonl $H/q_gemini_stats/q_gemini_stats_platt.jsonl
rep S_qe7_vs_mind q_mind_nonight $(j q_mind_nonight) $(j q_gemini) $(j q_gemini_stats) \
  $H/q_gemini/q_gemini_platt.jsonl $H/q_gemini_stats/q_gemini_stats_platt.jsonl
rep S_indep i_online_lr $(j i_online_lr) $(j i_population) $(j i_counter) $(j i_surprise_lr) $(j i_zeroshot) $(j i_p2)
rep S_indep_qe7 iq_online_lr $(j iq_online_lr) $(j iq_p2) $(j iq_gemini) $(j iq_gemini_stats) \
  $H/iq_gemini/iq_gemini_platt.jsonl $H/iq_gemini_stats/iq_gemini_stats_platt.jsonl
rep S_indep_qe7_vs_p2 iq_p2 $(j iq_p2) $(j iq_gemini) $(j iq_gemini_stats) \
  $H/iq_gemini/iq_gemini_platt.jsonl $H/iq_gemini_stats/iq_gemini_stats_platt.jsonl
$PY $T/t2boot.py p3 $(j m_mind_nonight) $(j m_online_lr) 1000 > $O/P3.json || { echo "FAILED P3"; exit 1; }
$PY $T/t2boot.py k1 $(j m_mind_k1) $(j m_mind) 1000 > $O/P6_k1.json || { echo "FAILED k1"; exit 1; }
$PY $T/t2boot.py p3 $(j i_p2) $(j i_online_lr) 1000 > $O/S_indep_P3.json || { echo "FAILED indep P3"; exit 1; }
$PY $T/endpoints.py $O > $O/SCORES.json.tmp || { echo "FAILED endpoints"; exit 1; }
mv $O/SCORES.json.tmp $O/SCORES.json
cat $O/SCORES.json; shasum -a 256 $O/SCORES.json
