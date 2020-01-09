#!/usr/bin/python3

import sys,fileinput, csv

db = 0

csv_field_names = None
csv_table=[]
for line in fileinput.input(sys.argv[1:]):
    if line.startswith("run,"):
        csv_field_names = line.replace(' ', '').split(',')
    elif csv_field_names and "Contraction" in line:
        csv_table.append(line)
        if db:
            print ("L=",line),

dictReader = csv.DictReader(csv_table, csv_field_names, skipinitialspace=True)

fieldsToKeep= ('problem-sizes', 'solution', 'time-us', 'gflops')

maxLen={}
for k in fieldsToKeep:
    maxLen[k] = 0

dataOut=[]
for row in dictReader:
    if db:
        print("R=", (row["solution"], row["time-us"], row["gflops"]))

    keep = {}
    for k in fieldsToKeep:
        keep[k] = row[k]
        maxLen[k] = max(len(row[k]), maxLen[k])

    dataOut.append((keep))

# sort by gflops
dataOut.sort(key = lambda row: int(row["gflops"]), reverse=True)

for row in dataOut:
    print (('{0: <%d}'%maxLen['solution']).format(row['solution']), " %6.1f %6d" %(float(row['time-us']), int(row['gflops'])))
