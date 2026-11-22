#!/bin/bash
# Work-server watcher (2026-11-20): when dstack task b1-tiny-dcae is no longer running (and its tiny weights exist or it ended),
# submit dstack/task_b23_sana.yaml. Keeps the 2-GPU limit: the two tasks never run at the same time.
cd /mnt/project/kchat/duyhv/i2i-sticker-avatar
while true; do
  st=$(timeout 60 dstack ps -a 2>/dev/null | awk '$1=="b1-tiny-dcae"{print $0}' | head -1)
  if [ -n "$st" ] && ! echo "$st" | grep -qE " running | provisioning | pulling | submitted | terminating "; then
    echo "B1 finished: $st $(date -Is)"; break
  fi
  sleep 120
done
ls -d weights/tiny_dcae/tdec1r/final weights/tiny_dcae/tenc1r/final 2>&1
timeout 300 dstack apply -f dstack/task_b23_sana.yaml -y -d 2>&1 | grep -v -i token | tail -3
echo "B23_SUBMITTED $(date -Is)"
