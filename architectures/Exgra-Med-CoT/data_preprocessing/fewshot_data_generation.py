import os
import argparse
import json
import random
def parse_option():
    parser = argparse.ArgumentParser('Processing few-shot dataset', add_help=False)
    parser.add_argument('--path', type=str, default="", help='Path to data file, e.g /netscratch/trnguyen/data_RAD/train_w_options_new.json', )
    parser.add_argument('--fewshot_size', type= int, choices=[50, 20, 10, 5, 1], default = 10, help='50, 20 or 10 percent of full dataset', )
    args = parser.parse_args()
    return args
args = parse_option()

with open(args.path, 'r') as f:
    data = json.load(f)
full_size = len(data)
fewshot_size = int(full_size*args.fewshot_size/100)

fewshot_file = os.path.splitext(os.path.basename(args.path))[0]
with open(os.path.join(os.path.dirname(args.path), fewshot_file + str(args.fewshot_size) + '.json'), 'w') as f: 
    fewshot_data = random.sample(data, fewshot_size)
    json.dump(fewshot_data, f)
