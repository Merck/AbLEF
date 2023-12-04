#!/bin/bash
for f in *.pdb
do
  subdir=${f%%_*}
  [ ! -d "$subdir" ] && mkdir -- "$subdir"
  mv -- "$f" "$subdir"
done
