#!/bin/bash
# Start the Deep ALM web app.
#   Put the "webapp" folder inside your project folder ("Research Internship FEB"),
#   next to the layer folders, then run:   bash webapp/start_app.sh
set -e
APP="$(cd "$(dirname "$0")" && pwd)"
BASE="$(dirname "$APP")"

# every layer folder that holds .py files goes on the Python path
PP="$APP"
for d in "$BASE"/*/; do
  d="${d%/}"
  [ "$d" = "$APP" ] && continue
  ls "$d"/*.py >/dev/null 2>&1 && PP="$PP:$d"
done
export PYTHONPATH="$PP"

# where your command-line results are, for the first run (only used once)
if [ -z "$ALM_FIT_DIR" ]; then
  f=$(find "$BASE" -name fit_sqrt.npz -not -path "*/webapp/*" 2>/dev/null | sort -r | grep -m1 output_paper || find "$BASE" -name fit_sqrt.npz -not -path "*/webapp/*" 2>/dev/null | head -1)
  [ -n "$f" ] && export ALM_FIT_DIR="$(dirname "$f")"
fi
if [ -z "$ALM_NETWORK_DIR" ]; then
  f=$(find "$BASE" -name deep_alm_DeepALM.pkl -not -path "*/webapp/*" 2>/dev/null | sort -r | grep -m1 output_paper || find "$BASE" -name deep_alm_DeepALM.pkl -not -path "*/webapp/*" 2>/dev/null | head -1)
  [ -n "$f" ] && export ALM_NETWORK_DIR="$(dirname "$f")"
fi
echo "Curve results:   ${ALM_FIT_DIR:-not found}"
echo "Network results: ${ALM_NETWORK_DIR:-not found}"

cd "$APP"
exec streamlit run app.py --browser.gatherUsageStats false
