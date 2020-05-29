#!/bin/bash
cat $1 | sort | uniq -c | awk '{printf("%s\n",$0)}'| awk '{$1=""}1' &>unique-$1
echo cat unique-$1
cat unique-$1
