#!/bin/bash
# Fleet-lite watcher for the canonical-116 L m=3 census.
# Emits one line whenever either pod's completed-item count changes, a
# 20-minute ETA summary, and a final banner when both pods report SHARDS DONE.
M4LOG="$HOME/code/adinkra-codespace/results/cls_g_csp_shards_L_3blocks_canonical_macm4.log"
SBSSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 brandon@192.168.68.71"
SBLOG='$HOME/adinkra-codespace-itemspec/results/cls_g_csp_shards_L_3blocks_canonical_stonkbot.log'
PROJECTED_TOTAL_NODES=374000000
prev=""
i=0
while true; do
  i=$((i + 1))
  m4hb=$(tail -40 "$M4LOG" 2>/dev/null | grep -E '^\[hb' | tail -1)
  sbhb=$($SBSSH "tail -40 $SBLOG" 2>/dev/null | grep -E '^\[hb' | tail -1)
  m4d=$(printf '%s' "$m4hb" | sed -n 's/.*items \([0-9]*\)\/[0-9]* done.*/\1/p')
  sbd=$(printf '%s' "$sbhb" | sed -n 's/.*items \([0-9]*\)\/[0-9]* done.*/\1/p')
  m4n=$(printf '%s' "$m4hb" | sed -n 's/.*nodes=\([0-9]*\).*/\1/p')
  sbn=$(printf '%s' "$sbhb" | sed -n 's/.*nodes=\([0-9]*\).*/\1/p')
  m4r=$(printf '%s' "$m4hb" | sed -n 's/.* \([0-9]*\)\/s avg.*/\1/p')
  sbr=$(printf '%s' "$sbhb" | sed -n 's/.* \([0-9]*\)\/s avg.*/\1/p')
  cur="m4=${m4d:-?} sb=${sbd:-?}"
  if [ "$cur" != "$prev" ]; then
    echo "progress macm4 ${m4d:-?}/29 items, stonkbot ${sbd:-?}/85 items"
    prev="$cur"
  fi
  if [ $((i % 20)) -eq 0 ]; then
    done_total=$(( ${m4d:-0} + ${sbd:-0} ))
    meas=$(( ${m4n:-0} + ${sbn:-0} ))
    rate=$(( ${m4r:-0} + ${sbr:-0} ))
    if [ "$rate" -gt 0 ] 2>/dev/null; then
      remain=$(( PROJECTED_TOTAL_NODES - meas - 14800000 ))
      if [ "$remain" -lt 0 ]; then remain=0; fi
      eta_s=$(( remain / rate ))
      eta_h=$(python3 -c "print(f'{$eta_s/3600:.1f}')" 2>/dev/null || echo "?")
      echo "summary ${done_total}/114 shards, ${meas} nodes measured, ${rate} n/s combined, eta ~${eta_h}h"
    fi
  fi
  m4done=$(grep -c "SHARDS DONE" "$M4LOG" 2>/dev/null)
  sbdone=$($SBSSH "grep -c 'SHARDS DONE' $SBLOG" 2>/dev/null || echo 0)
  if [ "${m4done:-0}" -ge 1 ] && [ "${sbdone:-0}" -ge 1 ]; then
    echo "BOTH PODS DONE: macm4 + stonkbot finished their canonical items"
    break
  fi
  sleep 60
done
