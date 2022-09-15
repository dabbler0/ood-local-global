import argparse
import nltk
import os
import torch
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()

files = os.listdir(args.input)

with open(args.output, 'w') as output:
    for filename in files:
        tree = ET.parse(os.path.join(args.input, filename))
        root = tree.getroot()
        for sentence in root.iter('s'):
            output.write(sentence.text + '\n')
