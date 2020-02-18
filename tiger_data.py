
'''
This Class reads the File containing annotated Data.
The Sentences and Tags are extracted from the CONLL09-Format
and then added to either Train-, Development- or Test-Data-Set.
Simultaneously, the Words (+count) and Tags are saved
for later processing (FOFE Encoding).
'''

from collections import defaultdict

############# Class Tiger_Data #############

class Tiger_Data():

	# Create Train-, Development- and Test-Data
	def __init__(self, filename):
		self.train_data = ([], [])
		self.dev_data = ([], [])
		self.test_data = ([], [])
		self.word_count = defaultdict(int)
		self.tag_set = set()
		n = 0
		# Call Generator and divide Sentences into Test-, Dev- and Train-Data
		for sentence, tags in self.read_tiger_data(filename):
			if n < 4000:
				self.test_data[0].append(sentence)
				self.test_data[1].append(tags)
			elif n in range(4000, 8000):
				self.dev_data[0].append(sentence)
				self.dev_data[1].append(tags)
			else:
				self.train_data[0].append(sentence)
				self.train_data[1].append(tags)
			n += 1

	# Read the CONLL09-Format & extract Words and Tags
	def read_tiger_data(self, filename):
		sentence = []
		tags = []
		with open(filename) as f:
			for line in f:
				if line.rstrip():
					lst = line.split('\t')
					word = lst[1]	# Extract Word from line
					tag = lst[4]	# Extract Tag
					sentence.append(word)
					tags.append(tag)
					self.word_count[word] += 1
					self.tag_set.add(tag)
				else:
					yield sentence, tags
					sentence = []
					tags = []
			if sentence:
				yield sentence, tags

	
