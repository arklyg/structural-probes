""" Demos a trained structural probe by making parsing predictions on stdin """

from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
import sys
import re
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

import data
import model
import probe
import regimen
import reporter
import task
import loss
import bin_tree
import run_experiment

from pytorch_pretrained_bert import BertTokenizer, BertModel

def report_on_file(args, src_path, trg_path, pickle_path):
  """Runs a trained structural probe on sentences piped to stdin.

  Sentences should be space-tokenized.
  A single distance image and depth image will be printed for each line of stdin.

  Args:
    args: the yaml config dictionary
  """

  # Define the BERT model and tokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
  model = BertModel.from_pretrained('bert-large-cased')
  LAYER_COUNT = 24
  FEATURE_COUNT = 1024
  model.to(args['device'])
  model.eval()

  # Define the distance probe
  distance_probe = probe.TwoWordPSDProbe(args)
  distance_probe.load_state_dict(torch.load(args['probe']['distance_params_path'], map_location=args['device']))

  # Define the depth probe
  depth_probe = probe.OneWordPSDProbe(args)
  depth_probe.load_state_dict(torch.load(args['probe']['depth_params_path'], map_location=args['device']))

  # Open output file
  src_lines = open(src_path).read().strip().split('\n')
  trg_lines = open(trg_path).read().strip().split('\n')
  pickle_file = open(pickle_path, 'wb')
  pairs = []

  #for index, line in tqdm(trg_lines, desc='[demoing]'):
  for index, line in enumerate(trg_lines):
    trg_line = line[: -1]
    if re.search("{\?!,\.}", trg_line):
      continue
    # Tokenize the sentence and create tensor inputs to BERT
    untokenized_sent = trg_line.strip().split()
    tokenized_sent = tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(trg_line.strip().split()) + ' [SEP]')
    #print(tokenized_sent)
    untok_tok_mapping = data.SubwordDataset.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)
    #print(untok_tok_mapping)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    #print(indexed_tokens)
    segment_ids = [1 for x in tokenized_sent]
    #print(segment_ids)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])

    tokens_tensor = tokens_tensor.to(args['device'])
    segments_tensors = segments_tensors.to(args['device'])

    with torch.no_grad():
      # Run sentence tensor through BERT after averaging subwords for each token
      encoded_layers, _ = model(tokens_tensor, segments_tensors)
      single_layer_features = encoded_layers[args['model']['model_layer']]
      representation = torch.stack([torch.mean(single_layer_features[0,untok_tok_mapping[i][0]:untok_tok_mapping[i][-1]+1,:], dim=0) for i in range(len(untokenized_sent))], dim=0)
      representation = representation.view(1, *representation.size())

      # Run BERT token vectors through the trained probes
      #distance_predictions = distance_probe(representation.to(args['device'])).detach().cpu()[0][:len(untokenized_sent),:len(untokenized_sent)].numpy()
      depth_predictions = depth_probe(representation).detach().cpu()[0][:len(untokenized_sent)].numpy()
      #print('\ndistance_predictions[', len(distance_predictions), ']:')
      #for i in range(len(untokenized_sent)):
      #  print(untokenized_sent[i], distance_predictions[i])
      #print('depth_predictions[', len(depth_predictions), ']:')
      #for i in range(len(untokenized_sent)):
      #  print(untokenized_sent[i], depth_predictions[i])

      word_depth = []
      for i in range(len(untokenized_sent)):
        word_depth.append((untokenized_sent[i], depth_predictions[i]))

      pairs.append({'src' : src_lines[index], 'trg' : find_top(word_depth, 0, len(word_depth))})
      print(index, trg_line)

  pickle.dump(pairs, pickle_file)
  pickle_file.close()


def find_top(segs, start, end):
  # end is excluded
  top_i, top_v = 0, sys.float_info.max
  for i in range(start, end):
    (_, cur_v) = segs[i]
    if cur_v < top_v:
      top_v = cur_v
      top_i = i
  
  if start < top_i:
    left = find_top(segs, start, top_i)
  else:
    left = None
  if top_i + 1 < end:
    right = find_top(segs, top_i + 1, end)
  else:
    right = None
  (data, _) = segs[top_i]
  return bin_tree.BinTreeNode(data, left, right)

if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('experiment_config')
  argp.add_argument('--results-dir', default='',
      help='Set to reuse an old results dir; '
      'if left empty, new directory is created')
  argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  argp.add_argument('--src_path',
      help='src file')
  argp.add_argument('--trg_path',
      help='trg file')
  argp.add_argument('--pickle', default='',
      help='pickle file')
  cli_args = argp.parse_args()
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  yaml_args= yaml.full_load(open(cli_args.experiment_config))
  run_experiment.setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  report_on_file(yaml_args, cli_args.src_path, cli_args.trg_path, cli_args.pickle)
