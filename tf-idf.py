# Set your index number here
admission_no = '140701M'

## Imports

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from math import log
import pandas as pd

## Reading documents

files = []
for i in range(1, 4):
    file = open('document/document' + str(i) + '.txt', 'r')
    files.append(file)

## Raw Tokens
# 1. Splitting into tokens by whitespaces

raw_token_set = []
for i in range(3):
    tokens = wordpunct_tokenize(files[i].read())
    raw_token_set.append(tokens)

# 2. Change all tokens into lowercase for accurate token comparison

processed_token_set = []
for i in range(3):
    processed_tokens = []
    for token in raw_token_set[i]:
        processed_tokens.append(token.lower())
    processed_token_set.append(processed_tokens)

## Terms
# * Removing duplicate tokens

terms = []
for i in range(3):
    terms.append(sorted(list(set(processed_token_set[i]))))

# * Accumulating all terms in all documents by removing duplicates

collection_terms = []
for i in range(3):
    collection_terms += terms[i]
collection_terms = sorted(list(set(collection_terms)))

## Statistical Calculations
# 1. Raw term frequencies and weighted log term frequencies

raw_tf = []
weighted_tf = []
for i in range(3):
    raw_frequency = []
    weighted_frequency = []
    for term in collection_terms:
        raw_count = processed_token_set[i].count(term)
        if(raw_count != 0):
            weighted_count = 1 + log(raw_count, 10)
        else:
            weighted_count = 0
        raw_frequency.append(raw_count)
        weighted_frequency.append(weighted_count)
    raw_tf.append(raw_frequency)
    weighted_tf.append(weighted_frequency)

# 2. Document frequencies and inverted document frequencies

doc_freq = []
invtd_doc_freq = []
for term in collection_terms:
    doc_count = 0;
    for i in range(3):
        if(term in terms[i]):
            doc_count += 1
    doc_freq.append(doc_count)
    invtd_doc_freq.append(log(3/doc_count, 10))

# 3. TF-IDF scores

tf_idf_scores = []
for i in range(3):
    tf_idf_score = []
    for j in range(len(collection_terms)):
        word = collection_terms[j]
        tf = weighted_tf[i][j]
        idf = invtd_doc_freq[j]
        tf_idf = tf*idf
        tf_idf_score.append(tf_idf)
    tf_idf_scores.append(tf_idf_score)

## Summarizing into a Dataframe
# 1. Create separate dataframes for each vocabulary

collection_stat = []
for i in range(3):
    collection_stat.append(pd.DataFrame({'term': collection_terms,
                                         'raw_term_frequnecy': raw_tf[i],
                                         'weighted_log_frequency': weighted_tf[i],
                                         'document_frequency': doc_freq,
                                         'inverse_document_frequency': invtd_doc_freq,
                                         'tf_idf_score': tf_idf_scores[i]}))

# 2. Removing all non occuring terms in each document

for i in range(3):
    collection_stat[i] = collection_stat[i][collection_stat[i].raw_term_frequnecy != 0]

### Document 1 - Summary

# collection_stat[0].head()

### Document 2 - Summary

# collection_stat[1].head()

### Document 3 - Summary

# collection_stat[2].head()

## Answers
### Question 1
# Find the vocabulary size  for given documents.

answer1 = ''
for i in range(3):
    answer = 'document%d:%d' % (i + 1, collection_stat[i].shape[0])
    print(answer)
    if(i < 2):
        answer1 += answer + '\n'
    else:
        answer1 += answer

### Question 2
# Find the TF value for all alphabetically ordered vocabulary words. Get the last two digits of your index and give words and corresponding TF values indexed in the places represented by last two digits. For example if your index is 140022X, then you need to give the words and corresponding TF value indexed at 22nd place in the vocabulary.

# * My index - 1407**01**M, therefore term indexed at 1st position in each document when terms in sorted order.

index = int(admission_no[4:6])

answer2 = ''
for i in range(3):
    answer = answer = 'document%d:%s,%.3f' % (i + 1, collection_stat[i].iloc[index, 0], collection_stat[i].iloc[index, 2])
    print(answer)
    if(i < 2):
        answer2 += answer + '\n'
    else:
        answer2 += answer

### Question 3
# Find the IDF value for all alphabetically ordered vocabulary words. Get the last two digits of your index and give words and corresponding IDF values indexed in the places represented by last two digits. For example if your index is 140022X, then you need to give the words and corresponding IDF value indexed at 22nd place in the vocabulary.

# * My index - 1407**01**M, therefore term indexed at 1st position in each document when terms in sorted order.

answer3 = ''
for i in range(3):
    answer = 'document%d:%s,%.3f' % (i + 1, collection_stat[i].iloc[index,0], collection_stat[i].iloc[index,4]) 
    print(answer)
    if(i < 2):
        answer3 += answer + '\n'
    else:
        answer3 += answer

### Question 4
# Find and report top 10 word for each document based on TF-IDF values.

word_set = []
for i in range(3):
    words = list(collection_stat[i].sort_values('tf_idf_score', ascending=False).head(10)['term'])
    word_set.append(words)

answer4 = ''
for i in range(3):
    answer_a = 'document%d:' % (i + 1)
    print(answer_a, end='')
    answer4 += answer_a
    for j in range(10):
        answer_b = word_set[i][j]
        if(j < 9):
            print(answer_b, end=',')
            answer4 += answer_b + ','
        else:
            print(answer_b)
            if(i < 2):
                answer4 += answer_b + '\n'
            else:
                answer4 += answer_b

## Saving Answers
# * Write into a text file

answers = open('answers.txt', 'w')
answers.write(admission_no + '\n')
answers.write('1' + '\n')
answers.write(answer1 + '\n\n')
answers.write('2' + '\n')
answers.write(answer2 + '\n\n')
answers.write('3' + '\n')
answers.write(answer3 + '\n\n')
answers.write('4' + '\n')
answers.write(answer4)
answers.close()