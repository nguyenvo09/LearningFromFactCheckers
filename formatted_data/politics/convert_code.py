import json
import collections


from typing import List
import os
from tqdm import tqdm
import pandas as pd
import itertools


def convert_json_to_tsv(infile: str, outfile: str):
    js = json.loads(open(infile).read())
    rows = []
    for row in tqdm(js):
        queryID = row["OriginalTweetID"]
        queryText = row["otweet_content"]
        docID = row["DTweetID"]
        docText = row["dtweet_content"]
        assert "\t" not in queryText
        assert "\t" not in docText
        rows.append([queryID, queryText, docID, docText])

    df = pd.DataFrame(rows, columns=["QueryID", "QueryText", "DocID", "DocText"])
    df.to_csv(outfile, index=False, sep="\t")


def convert_vocab(infile, outfile: str):

    fin = open(infile, "r")
    mapper = {}
    counter = itertools.count()
    for line in fin:
        line = line.replace("\n", "")
        mapper[line] = next(counter)

    s = json.dumps(mapper, sort_keys=True, indent=2)
    fout = open(outfile, "w")
    fout.write("%s\n" % s)


if __name__ == '__main__':
    # infile = "../data/train.json"
    # outfile = "mapped_data/sigir19.train.tsv"
    #
    # infile = "../data/val.json"
    # outfile = "mapped_data/sigir19.dev.tsv"

    # infile = "../data/test.json"
    # outfile = "mapped_data/sigir19.test.tsv"
    #
    # convert_json_to_tsv(infile, outfile)

    convert_vocab(infile="raw_data/vocab", outfile="mapped_data/vocab.json")
