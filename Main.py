import errno
import os
from os import listdir
from os.path import isfile, join
from nltk.corpus import brown
from nltk.tag import DefaultTagger
from pickle import dump
from pickle import load
from nltk.corpus import stopwords
from collections import Counter
import nltk
from Ontology import *
import re


def main():
    # Set up corpus reader for untagged data
    seminars_untagged_path = 'C:/Users/Gigabyte/AppData/Roaming/nltk_data/corpora/seminars_testdata/test_untagged'
    seminars_untagged_files = [f for f in listdir(seminars_untagged_path) if isfile(join(seminars_untagged_path, f))]
    corpus_untagged = nltk.corpus.reader.plaintext.PlaintextCorpusReader(seminars_untagged_path,
                                                                         seminars_untagged_files)

    # Set up corpus reader for tagged data
    seminars_tagged_path = 'C:/Users/Gigabyte/AppData/Roaming/nltk_data/corpora/seminars_testdata/test_tagged'
    seminars_tagged_files = [f for f in listdir(seminars_tagged_path) if isfile(join(seminars_tagged_path, f))]
    corpus_tagged = nltk.corpus.reader.plaintext.PlaintextCorpusReader(seminars_tagged_path,
                                                                       seminars_tagged_files)

    # Part 1 of assignment: entity tagging
    entity_tagging(corpus_tagged, corpus_untagged, seminars_untagged_files)

    # Part 2 of assignment: ontology construction
    ontology_construction(corpus_untagged, seminars_untagged_files)


# Tag files
def entity_tagging(corpus_tagged, corpus_untagged, seminars_untagged_files):
    # Construct a list of names to pass to the extract speaker method later
    first_names = []
    last_names = []

    print("Loading names for speaker recognition...")

    with open('C:/Users/Gigabyte/AppData/Roaming/nltk_data/names.family') as file:
        last_names = last_names + file.read().splitlines()
    with open('C:/Users/Gigabyte/AppData/Roaming/nltk_data/names.female') as file:
        first_names = first_names + file.read().splitlines()
    with open('C:/Users/Gigabyte/AppData/Roaming/nltk_data/names.male') as file:
        first_names = first_names + file.read().splitlines()

    print("Loaded names for speaker recognition!")

    # Declare regex for individual parts of the head
    location_tag_regex = '(?<=Place: ) *.*'
    speaker_tag_regex = '(?<=Who: )(?:(?!(?:,|\n|/|\()).)*'

    # Build up a list of all locations and speakers
    location_list = []
    speaker_list = []

    # See if stored in file, if not then grab all locations and speakers and store
    try:
        print("Trying to load saved lists of locations and speakers...")

        tags_input = open('location_list.pkl', 'rb')
        location_list = load(tags_input)
        tags_input = open('speaker_list.pkl', 'rb')
        speaker_list = load(tags_input)
        tags_input.close()
    except (OSError, IOError):
        print("No saved lists found, creating new ones...")
        for file in seminars_untagged_files:
            # Get place and speaker tags from the emails
            found_location = extract_header_tag(corpus_untagged.raw(file), location_tag_regex)
            found_speaker = extract_header_tag(corpus_untagged.raw(file), speaker_tag_regex)

            # Append results that aren't empty or repeats into the lists
            if found_location not in location_list and found_location is not None and found_location:
                location_list.append(found_location)

            if found_speaker not in speaker_list and found_speaker is not None and found_speaker:
                speaker_list.append(found_speaker)

        tag_output = open('location_list.pkl', 'wb')
        dump(location_list, tag_output, -1)
        tag_output = open('speaker_list.pkl', 'wb')
        dump(location_list, tag_output, -1)
        tag_output.close()

    # Print message to show progress
    print("Loaded!")

    # Totals for overall calculations at the end
    true_positives_classified_total = 0
    number_classified_total = 0
    true_positives_in_corpus_total = 0

    # Tag every file untagged file and store tags in emails dictionary
    for file in seminars_untagged_files:
        # Extract tags
        extracted_stime = find_stime(corpus_untagged.raw(file))
        extracted_etime = find_etime(corpus_untagged.raw(file))
        extracted_paragraph = extract_paragraphs(corpus_untagged.raw(file))
        extracted_sentence = extract_sentences(corpus_untagged.raw(file))
        extracted_speakers = extract_speaker(corpus_untagged.raw(file), first_names, last_names, speaker_list)
        extracted_locations = extract_location(corpus_untagged.raw(file), location_list)

        # Total of tagged entities
        all_self_tagged = extracted_stime + extracted_etime + extracted_paragraph + extracted_sentence + \
                          extracted_speakers + extracted_locations

        # Get all true positives in the corpus and  in corpus and concatenate into a list
        true_positives_in_email = extract_tagged_data(corpus_tagged.raw(file))

        true_positives_in_email_list = []

        for index, tags_list in true_positives_in_email.items():
            true_positives_in_email_list += tags_list

        # Make counters for self tagged and correctly tagged lists
        counter_all_self_tagged = Counter(all_self_tagged)
        counter_true_positives_in_email = Counter(true_positives_in_email_list)

        # Calculate amount of not found true positives
        untagged_true_positives = counter_true_positives_in_email - counter_all_self_tagged

        # Calculate amount of tagged true positives
        tagged_true_positives_total = len(true_positives_in_email_list) - sum(untagged_true_positives.values())

        # Check if 0 so division by 0 doesn't occur
        if len(all_self_tagged) != 0:
            # Calculate precision: true positives classified / number of classified
            precision = tagged_true_positives_total / len(all_self_tagged)

            # Calculate recall: true positives classified/true positives in corpus
            recall = tagged_true_positives_total / len(true_positives_in_email_list)

            # Calculate f1: 2 * (precision*recall))/(precision+recall))
            f1 = 2 * ((precision * recall) / (precision + recall))
        else:
            if tagged_true_positives_total == 0:
                precision = 1
                recall = 1
                f1 = 1
            else:
                precision = 0
                recall = 0
                f1 = 0

        # Add values for calculations for totals
        true_positives_classified_total += tagged_true_positives_total
        number_classified_total += len(all_self_tagged)
        true_positives_in_corpus_total += len(true_positives_in_email_list)

        # Print results
        print("For file: " + file)
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("f1: " + str(f1))
        print("-----------------------")

        # Write file
        save_file_with_tags(extracted_stime, extracted_etime, extracted_locations, extracted_speakers,
                            extracted_sentence, extracted_paragraph, file, corpus_tagged)

    # End of loop for each file, calculate overall values
    precision = true_positives_classified_total / number_classified_total
    recall = true_positives_classified_total / true_positives_in_corpus_total
    f1 = 2 * ((precision * recall) / (precision + recall))

    # Print results
    print("FOR ALL FILES")
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("f1: " + str(f1))
    print()


# Method for saving the tagged file
def save_file_with_tags(stime_tags, etime_tags, location_tags, speaker_tags, sentence_tags, paragraph_tags, file,
                        corpus_untagged):
    # Setify all tag lists and put into dictionary
    all_tags = dict()
    all_tags['paragraph'] = set(paragraph_tags)
    all_tags['sentence'] = set(sentence_tags)
    all_tags['stime'] = set(stime_tags)
    all_tags['etime'] = set(etime_tags)
    all_tags['location'] = set(location_tags)
    all_tags['speaker'] = set(speaker_tags)

    # Get email text
    email_text = corpus_untagged.raw(file)

    # Loop through all text and tag
    for tag_type in all_tags:
        for tagged_word in tag_type:
            if tagged_word in email_text:
                email_text.replace(tagged_word, '<' + tag_type + '>' + tagged_word + '</' + tag_type + '>')

    # Make output directory if it doesn't exist
    filename = 'C:/[CS Work]/NLP/Assignment/output/' + file

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, "w") as f:
        f.write(email_text)


# Extract all the tagged data and return as a dictionary
def extract_tagged_data(text):
    tag_list = ['stime', 'etime', 'location', 'speaker', 'sentence', 'paragraph']

    extracted_entities = dict()

    for item in tag_list:
        # Extract bits between tags
        regex = r'(?:<' + item + r'>)([\s\S]*?)(?:</' + item + r'>)'
        extracted_entities[item] = re.findall(regex, text)

        # Remove tags
        text = re.sub(r'<' + item + r'>|</' + item + r'>', "", text)

    return extracted_entities


# Remove labels and return the text
def remove_labels(text):
    return re.sub(r'<[a-zA-Z]+?>|</[a-zA-Z]+?>', "", text)


# Find all end times
def find_etime(text):
    # Get etime objects to iterate over
    etime_iter = re.finditer(
        r'((?<=-)|(?<=- )|(?<=to )|(?<=until ))((0\d|1\d|2[0-3]|\d)((:[0-5]\d)|(\s*[AaPp]\.?[mM]\.?))+)', text)

    # Iterate over etime to only extract relevant matches
    etime_list = []

    for time in etime_iter:
        etime_list.append(time.group(0))

    return etime_list


# Find all start times
def find_stime(text):
    # Extract stime to iterate over
    stime_iter = re.finditer(
        r'(?<!-)(?<!- )(?<=\s)((0\d|1\d|2[0-3]|\d)((:[0-5]\d)|(\s*[AaPp]\.?[mM]\.?))+)(?! from)',
        text)
    # Extract only relevant part, not whole match
    stime_list = []

    for time in stime_iter:
        stime_list.append(time.group(0))

    # These are all potential stimes in the list - find the most common, that's probably the start time
    stime_counter = Counter()

    # Split on ':' and ' ' to get the hour part - the most common one of these is the start time
    for time in stime_list:
        split_time = re.split(r'[: ]', time).pop(0)
        stime_counter[split_time] += 1

    # Check which times are the most popular ones and add those to the actual stimes
    final_stime_list = []
    for time in stime_list:
        if re.split('[: ]', time).pop(0) == stime_counter.most_common()[0][0]:
            final_stime_list.append(time)

    return final_stime_list


# Extract abstract - everything after "Abstract:"
def extract_abstract(text):
    extracted_abstract = re.findall(r'(?s)(?<=Abstract:)(?:.*$)', text)

    # Check if abstract isn't empty
    if extracted_abstract:
        return extracted_abstract.pop(0)
    else:
        return []


# Extract only the taggable parts of the file - no postedby or header
def extract_taggable(text):
    # Extract text until postedby line, excluding header
    until_postedby = re.findall(r'\n.*(?=PostedBy)', text, re.DOTALL)

    # If no postedby tag, return empty string
    if not until_postedby:
        return ""

    # Check if until posted by is a list
    if until_postedby:
        until_postedby = until_postedby.pop(0)
    else:
        until_postedby = ""

    # Get abstract
    abstract = extract_abstract(text)

    # Return concatenation of the two
    return until_postedby + abstract


# Extract paragraphs
def extract_paragraphs(text):
    if text:
        # Other regex (smarter but less successful): (?<=(?:\n\n))[^ ]{2,}.*?(?=(?:\n\n))

        # Extract paragraphs using regex - attempt 1
        extracted_paragraphs = re.findall(r'(?:\n\n).*?(?=\n\n)', extract_abstract(text), re.DOTALL)

        # If previous regex didn't work, try another
        if not extracted_paragraphs:
            extracted_paragraphs = re.findall(r'(?:\n\n).*', extract_abstract(text), re.DOTALL)

        # Get rid of leading new lines
        stripped_paragraphs = []

        for paragraph in extracted_paragraphs:
            all_leading_newlines_removed = False
            while not all_leading_newlines_removed:
                if paragraph[0] == '\n':
                    paragraph = paragraph[1:]
                else:
                    all_leading_newlines_removed = True

            stripped_paragraphs.append(paragraph)

        return stripped_paragraphs
    else:
        return []


# Extract the head (everything before abstract)
def extract_head(text):
    head = re.findall(r'.*(?=Abstract:)', text, re.DOTALL)
    if head:
        return head.pop(0)
    else:
        return None


# Extract sentences - THIS DOESNT GET RID OF WHITE SPACE BEFORE WORD
def extract_sentences(text):
    # Empty list to hold sentences
    sentences = []

    # Get paragraphs to iterate over
    paragraphs = extract_paragraphs(text)

    # Extract sentences from individual paragraphs
    for paragraph in paragraphs:
        extracted_sentence = re.findall(r'.*?(?=(?:\.))', paragraph, re.DOTALL)

        # Check if empty
        if extracted_sentence:
            sentences.append(extracted_sentence.pop(0))

    # Get rid of leading white spaces in sentences
    stripped_sentences = []

    for sentence in sentences:
        stripped_sentences.append(sentence.strip())

    return stripped_sentences


# Method to grab speaker
def extract_speaker(text, first_names, last_names, extracted_speaker_list):
    # Empty speakers list for all occurances of the speaker
    speakers = []

    # The found speaker
    speaker = None

    # Get "Place:" tag from head
    who_tag = extract_header_tag(text, '(?<=Who: )(?:(?!(?:,|\n|/|\()).)*')

    # If "Who:" tag exists, grab speaker from there
    if who_tag:
        # Append who tag to list of speakers
        speaker = who_tag

    # Try to find the speaker in the list of speakers
    if not speaker:
        speakers_from_list = extract_tag_from_list(text, extracted_speaker_list)

        if speakers_from_list:
            return speakers_from_list

    # If speaker in who tag, try tagging text
    if not speaker:
        # Load tagger if it is stored
        try:
            tagger_input = open('tagger.pkl', 'rb')
            tagger = load(tagger_input)
            tagger_input.close()
        # If not sored, declare and train new
        except (OSError, IOError):
            print("Tagger not found, creating one...")

            # Train sents
            train_sents = brown.tagged_sents()[:50000]

            # Declare backoff tagger for training
            tagger = backoff_tagger(train_sents, [nltk.UnigramTagger, nltk.BigramTagger, nltk.TrigramTagger],
                                    backoff=DefaultTagger('Untagged'))

            # Save tagger
            tagger_output = open('tagger.pkl', 'wb')
            dump(tagger, tagger_output, -1)
            tagger_output.close()

            print("Tagger created!")

        # Tag input with tagger
        tagged_words = []

        extracted_abstract = extract_abstract(text)

        if extracted_abstract:
            tagged_words = tagger.tag(nltk.word_tokenize(extract_abstract(text)))

        # Iterate over tags and see which could be a word - store tags in potential_names list
        potential_names = []

        # Iterate over every tagged word
        for index, word in enumerate(tagged_words):
            # Store parts that could be a name here from thus iteration
            temp_name = []

            new_index = index

            # If the first part is NP or untagged, check next
            if re.match(r'NP|Untagged', word[1]):
                # Add to temporary name
                temp_name.append(word)

                # Next index
                new_index = new_index + 1

                # Check if the new index is still within the range
                if new_index >= len(tagged_words):
                    break

                # While parts of a potential name are still found, add them to the potential name
                while re.match(r'NP|Untagged|\.|\"', tagged_words[new_index][1]):
                    temp_name.append(tagged_words[new_index])

                    # Next index
                    new_index = new_index + 1

                    # Check if the new index is still within the range
                    if new_index >= len(tagged_words):
                        break

            # If the temporary name isn't just a single word, add to the potential names list
            if len(temp_name) > 1:
                potential_names.append(temp_name)

        # Go through every temporary name, and if tagged untagged or NP, check against the loaded first and last names
        for potential_name in potential_names:
            # Set up new string to store name in
            temp_name = ""
            real_name = True

            first_name = True

            # Go through every word in the potential name and check if it's valid
            for word, tag in potential_name:
                if re.match(r'NP|Untagged', tag):
                    # If it's a first name, it has to be part of the first names list
                    if first_name:
                        if word in first_names:
                            temp_name = word
                            first_name = False
                        else:
                            real_name = False
                            break
                    # If not first name, has to be part of the last names list
                    else:
                        if word in last_names or word in first_names:
                            temp_name = temp_name + " " + word
                        else:
                            real_name = False
                            break
                # If it's not NP or untagged it's probably punctuation, so leave as is
                else:
                    # Has to be NP or untagged as first name!
                    if first_name:
                        real_name = False
                        break
                    else:
                        temp_name += word
                        first_name = False

            # If a real name has been found, stop the loop
            if real_name:
                speaker = temp_name
                break

    # If a speaker found, extract abstract and count how many times the place appears in it, adding to the total
    if speaker:
        abstract = extract_abstract(text)

        speakers.append(speaker)

        for n in range(abstract.count(speaker)):
            speakers.append(speaker)

        return speakers
    else:
        return []


# Method to extract certain tag from the header
def extract_header_tag(text, regex):
    final_tag = None

    # Get tag from head if it exits
    head = extract_head(text)
    if head:
        final_tag = re.findall(r'' + regex, head)

        if len(final_tag) > 0:
            final_tag = final_tag.pop(0).strip()

    return final_tag


# Extract tag from a given list of tags
def extract_tag_from_list(text, tag_list):
    # Variable which will hold the found tag
    found_tag = None

    # List of all found tags in the text
    found_tag_list = []

    # Extract only the taggable parts of the text - no postedby or header
    taggable_text = extract_taggable(text)

    # Split taggable parts by line
    if taggable_text:
        split_text = taggable_text.splitlines()
    else:
        return []

    # Check every tag in the list until you find one in the text
    for tag in tag_list:
        for line in split_text:
            # If a tag is found, stop searching and break the loop
            if tag in line:
                found_tag = tag
                break

        # Go through the text again finding all the mentions of the found tag
        if found_tag:
            for line in split_text:
                if found_tag in line:
                    found_tag_list.append(found_tag)
            break

    return found_tag_list


# Extract location from text
def extract_location(text, location_list):
    # Empty list to store locations
    locations = []

    # Get "Place:" tag from head if it exits
    head = extract_head(text)

    if head:
        place_tag = re.findall(r'(?<=Place: ) *.*', head)
    else:
        return []

    # If place tag exists, find it in the body and add to locations list
    if place_tag:
        place_tag = place_tag.pop(0).strip()

        # Append place tag to total locations
        locations.append(place_tag)

        # Extract abstract and count how many times the place appears in it, adding tot he total
        abstract = extract_abstract(text)

        for n in range(abstract.count(place_tag)):
            locations.append(place_tag)
    # If place tag doesn't exit, try to find it in the location list from other emails
    else:
        locations = extract_tag_from_list(text, location_list)

    return locations


# Backoff POS tagger
def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)

    return backoff


# Create ontology
def ontology_construction(corpus_untagged, seminars_untagged_files):
    print("Building ontology...")

    # Declare ontology
    ontology = Ontology()

    # Loop through all emails, extract their word frequencies and insert into ontology
    print("Inserting emails into ontology...")

    for file in seminars_untagged_files:
        # Print progress percentage
        print(str(int((int((file[0:3])) - 301) / len(seminars_untagged_files) * 100)) + '%')

        # Tokenize all words in the email
        tokenized_email = nltk.word_tokenize(extract_taggable(corpus_untagged.raw(file)).lower())

        # Split on full stops
        split_on_full_stop = []

        for word in tokenized_email:
            split_on_full_stop = split_on_full_stop + word.split('.')

        # Get rid of all stop words and lemmatize
        stop_words = set(stopwords.words('english'))
        wnl = WordNetLemmatizer()
        lemmatized_and_no_stop_words = []
        for word in split_on_full_stop:
            if word not in stop_words:
                lemmatized_and_no_stop_words.append(wnl.lemmatize(word))

        # Remove punctuation and numbers from email
        only_words = []
        for word in lemmatized_and_no_stop_words:
            if word.isalpha():
                only_words.append(word)

        # Count words
        word_counts = Counter(only_words)

        # Insert email into ontology
        ontology.insert_email(file, word_counts)

    # Print out ontology tree
    ontology.print_tree()


main()
