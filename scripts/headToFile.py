import pandas as pd
import argparse

# CLI arguments for input and output file paths and number of rows to sample
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/assessment-data.csv')
parser.add_argument('--output', type=str, default='data/assessment-data-sample.csv')
parser.add_argument('--rows', type=int, default=50)
args = parser.parse_args()

# Read the input file and sample the first n rows and write to the output file path
df = pd.read_csv(args.input)    
df.head(args.rows).to_csv(args.output, index=False)