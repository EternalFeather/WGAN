# -*- coding:utf-8 -*-
import numpy as np
import collections


class Data_loader(object):
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.num_batch = 0
		self.filtered_lines, self.lines_token = [], []
		self.charmap = {'unk': 0}
		self.inv_charmap = ['unk']
		self.lines_batch = np.array([])
		self.tokens_batch = np.array([])
		self.pointer = 0

	def load_datasets(self, max_length, example_num, vocab_size, data_path, tokenize=False):
		print("Loading datasets...")
		lines = []

		with open(data_path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip("\n")
				if tokenize:
					# word-based
					line = self.tokenize_string(line)
				else:
					# character-based
					line = tuple(line)

				if len(line) > max_length:
					line = line[: max_length]

				lines.append(line + (("`",)*(max_length-len(line))))    # padding with "`"
				if len(lines) == example_num:
					break

		np.random.shuffle(lines)
		counts = collections.Counter(char for line in lines for char in line)

		for char, count in counts.most_common(vocab_size-1):
			if char not in self.charmap:
				self.charmap[char] = len(self.inv_charmap)
				self.inv_charmap.append(char)

		for line in lines:
			filtered_line = []
			for char in line:
				if char in self.charmap:
					filtered_line.append(char)
				else:
					filtered_line.append('unk')
			self.filtered_lines.append(tuple(filtered_line))

		print("loaded {} lines in dataset".format(len(lines)))
		return self.filtered_lines, self.charmap, self.inv_charmap

	def tokenize_string(self, sentence):
		return tuple(sentence.lower().split(" "))

	def mini_batch(self):
		self.lines_token = [np.array([self.charmap[c] for c in l]) for l in self.filtered_lines]
		self.num_batch = int(len(self.filtered_lines) / self.batch_size)
		self.filtered_lines = self.filtered_lines[:self.num_batch * self.batch_size]
		self.lines_token = self.lines_token[:self.num_batch * self.batch_size]
		self.filtered_lines = np.array(self.filtered_lines)
		self.lines_token = np.array(self.lines_token)
		self.lines_batch = np.split(self.filtered_lines, self.num_batch, 0)
		self.tokens_batch = np.split(self.lines_token, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = self.tokens_batch[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0
