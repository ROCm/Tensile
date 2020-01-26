#!/usr/bin/python3

import sys,fileinput, csv

class Solution:
    def __init__(self):
        self.problemResults = []
        self.maxPerf = None

    def addRow(self,row):
        self.problemResults.append(row)
        if self.maxPerf==None or float(row["gflops"]) > float(self.maxPerf["gflops"]):
            self.maxPerf = row


class Reader:

    def formatCol(self, row, colName, val=None):
        return ('{0: <%d}'%self.maxLen[colName]).format(val if val else row[colName])


    def readFile(self):
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

        fieldsToPrint= ('problem-sizes', 'solution', 'time-us', 'gflops', "alu-us","mem-read-us","mem-write-us", 'total-granularity')
        derivedFields = ['best-any']

        self.maxLen={}
        for k in fieldsToPrint:
            self.maxLen[k] = len(k) # length of name

        self.dataOut=[]
        solutions = {} # map from solution to Solution class
        for row in dictReader:
            solutions.setdefault(row["solution"],Solution()).addRow(row)

            keep = {}
            for k in fieldsToPrint:
                keep[k] = row[k]
                self.maxLen[k] = max(len(row[k]), self.maxLen[k])

            self.dataOut.append((keep))

        # sort by gflops
        self.dataOut.sort(key = lambda row: int(row["gflops"]), reverse=True)

        # print header:
        for field in fieldsToPrint:
            print (self.formatCol(row,field,val=field), end=" ")
        for field in derivedFields:
            print (field,end=" ")
        print()

        for row in self.dataOut:
            for field in fieldsToPrint:
                print (self.formatCol(row,field), end=" ")
            print (solutions[row["solution"]].maxPerf["gflops"], end="") # max perf from any solution
            print()

            #print (self.formatCol(row,'problem-sizes'), self.formatCol(row, 'solution'), " %6.1f %6d" %(float(row['time-us']), int(row['gflops'])))


r = Reader()
r.readFile()
