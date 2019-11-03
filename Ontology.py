import gensim
from nltk.stem import WordNetLemmatizer


class Ontology:
    def __init__(self):
        # Declare shared topic words - there are better ways of doing this however for a small ontology like this it's
        # enough, otherwise I'd have to rewrite my ontology and all methods since the shared topic words would be
        # stored in "top" which are all leaf nodes and therefore can't be referenced from the next node easily
        self.cs_topic_words = ['computer', 'technology', 'speakers', 'programming', 'computing', 'system',
                               'intel', 'multimedia', 'telephony', 'cs']

        self.robotics_topic_words = self.cs_topic_words + ['robot', 'robotics']

        self.robotics_ai_topic_words = self.robotics_topic_words + ['d*', 'a*', 'route', 'plan', 'feedback', 'ai',
                                                                    'artificial', 'intelligence', 'algorithm',
                                                                    'recognition', 'neural', 'network', 'net',
                                                                    'planning', 'machine']

        # Declare ontology
        self.ontology = {
            'top': {
                'cs': {
                    'cs general': {
                        'topic_words': self.cs_topic_words,
                        'talks': list()
                    },
                    'ai': {
                        'topic_words': self.cs_topic_words + ['ai', 'artificial', 'intelligence', 'algorithm',
                                                              'learning', 'recognition', 'neural', 'network', 'net',
                                                              'planning', 'machine', 'learning', 'node', 'deep',
                                                              'feedback', 'plan', 'planning'],
                        'talks': list(),
                    },
                    'teaching': {
                        'topic_words': self.cs_topic_words + ['teaching', 'classroom', 'ta', 'tas', 'lectures',
                                                              'academic', 'assignments', 'course', 'graduate'],
                        'talks': list(),
                    },
                    'nlp': {
                        'topic_words': self.cs_topic_words + ['syntax', 'semantics', 'pragmatics', 'comprehension',
                                                              'language', 'translation', 'syntax', 'syntactic',
                                                              'parsing', 'speech', 'recognition'],
                        'talks': list(),
                    },
                    'software engineering': {
                        'topic_words': self.cs_topic_words + ['prototype', 'design', 'software', 'architecture',
                                                              'parallel', 'parallelization', 'bug', 'debug',
                                                              'debugging', 'system', 'requirements'],
                        'talks': list()
                    },
                    'computer systems': {
                        'topic_words': self.cs_topic_words + ['parallel', 'processor', 'memory', 'i/o', 'architecture',
                                                              'performance', 'thread', 'process', 'computer', 'state',
                                                              'machine', 'supercomputer', 'multiprocessors', 'dram',
                                                              'latency', 'cache', 'compiler'],
                        'talks': list(),
                    },
                    'product design': {
                        'topic_words': self.cs_topic_words + ['design', 'product', 'manufacture', 'manufacturability'],
                        'talks': list(),
                    },
                    'graphics': {
                        'topic_words': self.cs_topic_words + ['graphics', 'color', 'image', 'photo', 'vision',
                                                              'visual'],
                        'talks': list(),
                    },
                    'maths': {
                        'topic_words': self.cs_topic_words + ['math', 'mathematical', 'problem', 'geometric', 'linear',
                                                              'operators', 'matrices', 'function', 'linear', 'curves',
                                                              'algebra', 'equations', 'polynomial', 'polyhedra',
                                                              'complexity', 'arithmetic', 'mathematics', 'logic',
                                                              'square', 'root', 'graph'],
                        'talks': list(),
                    },
                    'internet': {
                        'topic_words': self.cs_topic_words + ['internet', 'web', 'information', 'browsers', 'indexing'],
                        'talks': list(),
                    },
                    'programming languages': {
                        'topic_words': self.cs_topic_words + ['object-oriented', 'language', 'functional', 'fortran'],
                        'talks': list(),
                    },
                    'robotics': {
                        'robotics general': {
                            'topic_words': self.robotics_topic_words,
                            'talks': list()
                        },
                        'robotics maths': {
                            'topic_words': self.robotics_topic_words + ['math', 'mathematical', 'problem', 'geometric',
                                                                        'linear', 'operators', 'matrices', 'function',
                                                                        'linear', 'curves', 'algebra', 'equations',
                                                                        'polynomial', 'polyhedra', 'complexity'],
                            'talks': list(),
                        },
                        'motion/sensors/motion': {
                            'topic_words': self.robotics_topic_words + ['motors', 'micromotors', 'control', 'walking',
                                                                        'sensor', 'sensing',
                                                                        'space'],
                            'talks': list(),
                        },
                        'robotics product design': {
                            'topic_words': self.robotics_topic_words + ['design', 'product', 'manufacture',
                                                                        'manufacturability'],
                            'talks': list(),
                        },
                        'vr': {
                            'topic_words': self.robotics_topic_words + ['vr', 'virtual', 'reality'],
                            'talks': list(),
                        },
                        'robotics ai': {
                            'robotics ai general': {
                                'topic_words': self.robotics_ai_topic_words,
                                'talks': list()
                            },
                            'robotics vision': {
                                'topic_words': self.robotics_ai_topic_words + ['image', 'vision', 'see', 'photo',
                                                                               'photogrammetry', 'color', 'resampling',
                                                                               'raster', 'visual', 'imagery'],
                                'talks': list(),
                            }
                        },
                        'exploration': {
                            'topic_words': self.robotics_topic_words + ['exploration', 'navigate', 'antarctic',
                                                                        'planetary', 'volcano'],
                            'talks': list(),
                        }
                    }
                },
                'science': {
                    'topic_words': ['biomechanics', 'biomolecules', 'molecular', 'biochemistry', 'molecule'],
                    'talks': list(),
                },
                'engineering': {
                    'topic_words': ['engineering', 'acceleration', 'force', 'friction', 'work', 'contact'],
                    'talks': list(),
                },
                'uncategorized': {
                    'topic_words': [],
                    'talks': list(),
                }
            }

        }

        # Load word2vec model
        print("Loading word2vec model...")
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            'C:/Users/Gigabyte/AppData/Roaming/nltk_data/GoogleNews-vectors-negative300.bin', binary=True)
        print("Loaded!")

        # Traversal info for easier retrieval of topic words
        self.traversal_info = dict()

        # Create traversal info
        self.create_traversal_info()

        # Optimise traversal info - stem
        self.optimise_traversal_info()

    # Create traversal info dictionary
    def create_traversal_info(self, location_list=None):
        # Set traversed location to start of the tree
        traversed_location = self.ontology

        # Check if the location list has any elements (not start of tree)
        if location_list is None:
            location_list = []

        # Make a copy of the location list to take bits out of and advance to correct location
        temp_location_list = location_list.copy()

        for node in location_list:
            if len(temp_location_list) > 0:
                traversed_location = traversed_location[node].copy()
                temp_location_list.pop(0)

        # Get keys from the destination location and traverse through them
        keys = traversed_location.keys()

        for key in keys:
            # If has topic words - leaf node, so extract related words ands return
            if key == 'topic_words':
                self.traversal_info[location_list[-1]] = [location_list, traversed_location['topic_words']]
                return

            # Not leaf node, call function on all child nodes
            new_location_list = location_list.copy()
            new_location_list.append(key)

            self.create_traversal_info(new_location_list)
        return

    # Optimise traversal info dictionary
    def optimise_traversal_info(self):
        # Iterate through traversal info dictionary
        for key, value in self.traversal_info.items():
            # Iterate through topic words of each node and lemmatize
            optimized_words = []

            for word in value[1]:
                wnl = WordNetLemmatizer()
                optimized_words.append(wnl.lemmatize(word))

            value[1] = optimized_words

    # Method for inserting emails into ontology
    def insert_email(self, file_name, word_frequencies):
        # Variables for keeping track of the current best node and score
        current_best_node = None
        current_best_score = 0

        # Iterate through traversal tree
        for node, value in self.traversal_info.items():
            # total score for this node
            total_score = 0
            # Iterate through topic words
            for topic_word in value[1]:
                # Iterate through the email words and their frequencies
                for email_word, frequency in word_frequencies.most_common():
                    try:
                        # If email word in topic words, take direct. Calculate score dependent on frequency and amount
                        # of topic words, multiplied by 1/5 as I have found this weight is very accurate
                        if email_word in value[1]:
                            total_score += frequency * (1 / len(value[1]*5))
                        # Otherwise use word2vec to get a similarity score and calculate as before
                        else:
                            if self.word2vec_model.similarity(topic_word, email_word) > 0.9:
                                total_score += self.word2vec_model.similarity(topic_word, email_word) * frequency * (
                                        1 / len(value[1]*5))
                    # If not in model or in list, do nothing
                    except KeyError:
                        pass

            # Check if this node is better than prior ones
            if total_score > current_best_score:
                current_best_score = total_score
                current_best_node = value[0]

        # If current best note not found, set as uncategorized
        if not current_best_node:
            current_best_node = ['top', 'uncategorized']

        # Go through ontology and insert the file name into "talks" list
        traversed_node = self.ontology

        for item in current_best_node:
            traversed_node = traversed_node[item]

        traversed_node['talks'].append(file_name)

    # Method for printing out the tree
    def print_tree(self, graph=None, indent=0):
        # If graph not specified i.e. first call
        if not graph:
            graph = self.ontology

        # Iterate through all keys and values
        for key, value in graph.items():
            # If a name of node, print
            if not key == 'topic_words' and not key == 'talks':
                print('----' * indent + str(key))
            # If another dictionary, call again with an increased indent
            if isinstance(value, dict):
                self.print_tree(value, indent + 1)
            # If a child node, print out the talks
            else:
                if key == 'talks':
                    print('    ' * indent + '└-->' + str(value))
