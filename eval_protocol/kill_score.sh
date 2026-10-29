#!/bin/bash
for p in $(pgrep -u $USER -f "score_mevis_login[.]sh"); do kill $p; done
for p in $(pgrep -u $USER -f "eval_mevis_keep[.]py"); do kill $p; done
sleep 2; echo "left: $(pgrep -u $USER -f "eval_mevis_keep[.]py" | wc -l)"
