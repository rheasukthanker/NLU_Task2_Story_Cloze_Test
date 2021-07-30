#!/usr/bin/env python2
from __future__ import print_function

import os
import numpy
import random
import csv
from os.path import join as pathjoin


DATA_PATH = 'data'
file_path_gpt = pathjoin(DATA_PATH, 'gpt_output.txt')
file_path_train_stories = pathjoin(DATA_PATH, 'train_stories.csv')
output_path = pathjoin(DATA_PATH, 'gpt_dataset.csv')
word_ending_characters = [".", "?", "!"]
endings = []
line_index = 0
wait_for_next_start_seq = False

with open(file_path_gpt) as f:
  for line in f:
    start_seq = "Model prompt >>>"

    if wait_for_next_start_seq and line[:len(start_seq)] != start_seq:
      continue

    if line[:len(start_seq)] == start_seq:
      wait_for_next_start_seq = False
      line_index += 1
      continue

    elif line[0] == "=" or line[:4] == "\r\n":
      continue

    else:
      # Handle line with actual content
      
      ending = ""
      for word in line:
        ending += word
        if word in word_ending_characters:
          break
      # Try next line if ending too short
      if len(line) > 2:
        endings.append([ending, line_index])
        wait_for_next_start_seq = True
      else:
        continue


print("\nNumber of endings: " + str(len(endings)) + "\n")

print("Example endings: ")
for i in range(10):
  idx = random.randint(0, len(endings)-1)
  #idx = i
  print(str(endings[idx]) + ", Length: " + str(len(endings[idx][0])))

max_idx = 0
for ending, idx in endings:
  max_idx = max(max_idx, idx)
print("\nMaximal line index in endings: " + str(max_idx))

output_file = open(output_path, mode='w')
   
with open(file_path_train_stories) as f:
  csv_reader = csv.reader(f, delimiter=',')
  tsv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  line_count = 0
  write_count = 0
  for row in csv_reader:
    if line_count == 0:
      header_row = ["InputStoryid","InputSentence1","InputSentence2","InputSentence3","InputSentence4",
                    "RandomFifthSentenceQuiz1","RandomFifthSentenceQuiz2","AnswerRightEnding"]
      tsv_writer.writerow(header_row)
      line_count += 1
    else:
      read_data = row
      gpt_ending = endings[line_count-1][0]
      line_index = endings[line_count-1][1]
      if line_count < 2:
        print(line_count)
        print(gpt_ending)
        print(line_index)
        print(read_data)
      if len(gpt_ending) > 3 and line_count == line_index:
        data_row = [read_data[0]] + read_data[2:7] + [gpt_ending] + ["1"]
        tsv_writer.writerow(data_row)
        write_count += 1
      line_count += 1
  print("Wrote dataset with " + str(write_count) + " lines.")
