#!/bin/bash
cd /scratch/ts1v23/workspace/noKslot_clean
CACHE=data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300
PY=/scratch/ts1v23/.conda/bin/python3
# wait until empirical_stats.pt exists OR the prewarm proc dies (whichever first)
until [ -f "$CACHE/empirical_stats.pt" ] || ! pgrep -u ts1v23 -f '[t]rain_graph_codeflow' >/dev/null 2>&1; do sleep 20; done
echo "=== empirical_stats.pt ==="; ls -la "$CACHE/empirical_stats.pt" 2>&1
if [ -f "$CACHE/empirical_stats.pt" ]; then
  $PY -c "import torch; s=torch.load('$CACHE/empirical_stats.pt',map_location='cpu'); print('count',s.get('count'),'D',s.get('D'),'mean',tuple(s['mean'].shape) if 'mean' in s else None,'std',tuple(s['std'].shape) if 'std' in s else None)" 2>&1
  echo "=== kill prewarm (scan done; free GPU for smoke) ==="
  PIDS=$(pgrep -u ts1v23 -f '[t]rain_graph_codeflow')
  [ -n "$PIDS" ] && kill $PIDS 2>/dev/null
  sleep 3
  echo "prewarm_procs=$(pgrep -u ts1v23 -fc '[t]rain_graph_codeflow')"
  echo EMPIRICAL_READY
else
  echo "PREWARM DIED before empirical_stats.pt — check log:"
  tail -10 scripts/_prewarm_v4b_empirical.log 2>/dev/null
  echo EMPIRICAL_FAILED
fi
