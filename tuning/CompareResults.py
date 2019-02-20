import os
import sys
import argparse
import re

import pandas as pd


def RunMain():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "current_file", help="path where the current results are located")
    argParser.add_argument(
        "new_file", help="path where the new files are located")
    argParser.add_argument(
        "combined_file", help="path where the combined results are located")

    args = argParser.parse_args(userArgs)

    currentFileName = args.current_file
    newFileName = args.new_file
    combinedFileName = args.combined_file

    current_data = pd.read_csv(currentFileName)
    headers = current_data.columns.values.tolist()
    key = headers[0:len(headers) - 1]

    current_data.set_index(key, inplace=True)
    new_data = pd.read_csv(newFileName)
    new_data.set_index(key, inplace=True)

    dfList = [current_data, new_data]
    result = pd.concat(
        dfList, keys=['current', 'new'], axis='columns', names=[None, None])

    result.to_csv(combinedFileName, header=True)


if __name__ == "__main__":
    RunMain()
