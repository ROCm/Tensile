#!/usr/bin/python3

import sys,fileinput, csv

class Reader:

    def formatCol(self, row, colName):
        return ('{0: <%d}'%self.maxLen[colName]).format(row[colName])

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

        fieldsToKeep= ('problem-sizes', 'solution', 'time-us', 'gflops')

        self.maxLen={}
        for k in fieldsToKeep:
            self.maxLen[k] = 0

        self.dataOut=[]
        for row in dictReader:
            if db:
                print("R=", (row["solution"], row["time-us"], row["gflops"]))

            keep = {}
            for k in fieldsToKeep:
                keep[k] = row[k]
                self.maxLen[k] = max(len(row[k]), self.maxLen[k])

            self.dataOut.append((keep))

        # sort by gflops
        self.dataOut.sort(key = lambda row: int(row["gflops"]), reverse=True)

        for row in self.dataOut:
            print (self.formatCol(row,'problem-sizes'), self.formatCol(row, 'solution'), " %6.1f %6d" %(float(row['time-us']), int(row['gflops'])))



r = Reader()
r.readFile()
