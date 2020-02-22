#!/usr/bin/python3

import sys,fileinput,csv
import argparse
from collections import OrderedDict

class Solution:
    """ Rows associated with this solution """
    def __init__(self):
        self.problemResults = []
        self.maxPerf = None

    def addRow(self,row):
        self.problemResults.append(row)
        if self.maxPerf==None or float(row["gflops"]) > float(self.maxPerf["gflops"]):
            self.maxPerf = row

class ProblemSize:
    """ Rows associated with this problem """
    def __init__(self):
        self.rows = []

    def addRow(self,row):
        self.rows.append(row)


class Reader:

    def formatCol(self, row, colName, val=None):
        return ('{0: <%d}'%self.maxLen[colName]).format(val if val else row[colName])

    def printRow(self, row, fieldsToPrint, end='\n'):
        for field in fieldsToPrint:
            print (self.formatCol(row,field), end=" ")
        print (end=end)


    def readFile(self, fileName):
        db = 0

        csv_field_names = None
        csv_table=[]
        for line in fileinput.input(fileName):
            if line.startswith("run,"):
                csv_field_names = line.replace(' ', '').split(',')
            elif csv_field_names and "Contraction" in line:
                csv_table.append(line)
                if db:
                    print ("L=",line),

        dictReader = csv.DictReader(csv_table, csv_field_names, skipinitialspace=True)

        fieldsToPrint= ('problem-progress', 'problem-sizes', 'solution', 'time-us', 'gflops', 'alu-us', 'mem-read-us', 'mem-write-us', 'tiles-per-cu', 'total-granularity', 'cu-granularity', 'tile0-granularity', 'tile1-granularity')
        derivedFields = ['best-any']

        self.maxLen={}
        for k in fieldsToPrint:
            self.maxLen[k] = len(k) # length of name

        self.dataOut=[]
        solutions = {} # map from solution to Solution class
        problemSizes = OrderedDict()
        for row in dictReader:
            solutions.setdefault(row["solution"],Solution()).addRow(row)
            problemSizes.setdefault(row["problem-sizes"],ProblemSize()).addRow(row)

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

        if args.print_winners:
            for key,ps in problemSizes.items():
                if args.problem_progress==None or int(ps.rows[0]["problem-progress"].split('/')[0]) in args.problem_progress:
                    sortedRows = sorted(ps.rows, key = lambda row: int(row["gflops"]), reverse=True)
                    for i in range(min(len(sortedRows),args.print_winners) if args.print_winners>=0 else len(sortedRows)):
                        self.printRow(sortedRows[i], fieldsToPrint,end='')
                        print (solutions[sortedRows[i]["solution"]].maxPerf["gflops"], end="") # max perf from any solution
                        print()

        if args.print_summary:
            for row in self.dataOut:
                self.printRow(row, fieldsToPrint, end='')
                print (solutions[row["solution"]].maxPerf["gflops"], end="") # max perf from any solution
                print()

            #print (self.formatCol(row,'problem-sizes'), self.formatCol(row, 'solution'), " %6.1f %6d" %(float(row['time-us']), int(row['gflops'])))



my_parser = argparse.ArgumentParser(description='summarize and analyze tensile benchmark results')

my_parser.add_argument('--print_winners', '-w', type=int, default=1,
                       help='print top N winners for each problem.  -1 prints all')
my_parser.add_argument('--problem_progress', '-p', type=int, action='append',
                       help='show only specified problem.  Can specify multiple times.')
my_parser.add_argument('--print_summary', '-s', action='store_true',
                       help='print all results')

my_parser.add_argument('input_file', help='file with tensile output to parse')

args = my_parser.parse_args()

r = Reader()
r.readFile(args.input_file)
