#!/bin/bash
# 10-minute watch of the census item-52 race (stonkbot T32 vs macm4 duplicate T12).
# Emits one depth line per cycle: each machine's nodes into item 52.
# Emits ITEM 52 DONE and exits the moment either side writes shard_0052.json.

M4_LOG=/Users/brandon/code/adinkra-codespace/results/cls_g_csp_shards_L_3blocks_canonical_macm4_item52.log
M4_DIR=/Users/brandon/code/adinkra-codespace/results/cls_g_csp_shards_L_3blocks_canonical
SSH_STONK="ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 brandon@192.168.68.71"
STONK_LOG='$HOME/adinkra-codespace-itemspec/results/cls_g_csp_shards_L_3blocks_canonical_stonkbot.log'
STONK_DIR='$HOME/adinkra-codespace-itemspec/results/cls_g_csp_shards_L_3blocks_canonical'
# stonkbot node total at the moment item 52 became its only working item
STONK_BASE=241776233

while true; do
  if [ -f "$M4_DIR/shard_0052.json" ]; then
    grep "shard 0052 done" "$M4_LOG" | tail -1
    echo "ITEM 52 DONE on macm4"
    exit 0
  fi
  if $SSH_STONK "[ -f $STONK_DIR/shard_0052.json ]" 2>/dev/null; then
    $SSH_STONK "grep -h 'shard 0052 done\|SHARDS DONE' $STONK_LOG | tail -2"
    echo "ITEM 52 DONE on stonkbot"
    exit 0
  fi
  m4_nodes=$(tail -1 "$M4_LOG" 2>/dev/null | sed -E 's/.*nodes=([0-9]+).*/\1/')
  st_line=$($SSH_STONK "tail -1 $STONK_LOG" 2>/dev/null)
  st_nodes=$(echo "$st_line" | sed -E 's/.*nodes=([0-9]+).*/\1/')
  if [ -n "$m4_nodes" ] && [ -n "$st_nodes" ]; then
    st52=$((st_nodes - STONK_BASE))
    echo "item52 race: macm4 ${m4_nodes} nodes, stonkbot ${st52} nodes into item 52"
  elif [ -z "$m4_nodes" ]; then
    echo "WARN: macm4 log unreadable or process gone (pid 54628)"
  else
    echo "WARN: stonkbot ssh or log unreadable"
  fi
  sleep 600
done
