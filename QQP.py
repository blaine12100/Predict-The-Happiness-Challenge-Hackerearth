import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import gensim
from nltk.corpus import stopwords
import nltk
import re
import string
from nltk.corpus import wordnet
from catboost import CatBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
'''def get_unigrams(question):
    return [word for word in word_tokenize(question.lower()) if word not in eng_stopwords]

def common_unigrams(question):#Perform Set operation on both unigrams
    return len(set(question["unigrams_quea1"]).intersection(set(question["unigrams_ques2"])))

train_df["unigrams_ques1"] = train_df['question1'].apply(lambda x: get_unigrams(str(x)))
train_df["unigrams_ques2"] = train_df['question2'].apply(lambda x: get_unigrams(str(x)))
'''

'''Opening directory path'''

path=os.path.normpath('M:\PycharmProjects\AI+DL+CP\QQP')

train_df=0
test_df=0

for subdir,dir,files in os.walk(path):
    for file in files:
        #print(file)
        if file =='QQPT.csv':
            train_df=pd.read_csv(os.path.join(subdir,file),encoding='utf-8')

        elif file=='QQPTest.csv':
            test_df=pd.read_csv(os.path.join(subdir,file),encoding='utf-8')

eng_stopwords = set(stopwords.words('english'))
#Null Data is not present
'''for index in range(len(train_dataset['question1'])):
    data=train_dataset.ix[index,['qid1','qid2','is_duplicate','question1','question2']]
    print(pd.isnull(data))

for roll in range(len(test_dataset['question1'])):
    sample=test_dataset.ix[roll,['question1','question2']]
    print(pd.isnull(sample))
'''


'''print(duplicate_number.index)
plt.figure(figsize=(10,5))
plt.bar(duplicate_number.index,duplicate_number.values,color='red',alpha=0.5)
plt.xlabel('Category')
plt.ylabel('Iterations')
plt.title('Catrgories vs Iterations')
#plt.legend()
plt.show()
'''
#duplicate_number=train_dataset['is_duplicate'].value_counts()
#print("Values",duplicate_number/duplicate_number.sum())
'''63% is non duplicate data while 37% is duplicate'''

print(eng_stopwords)

def remove_special_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))#Converts the special cahracters into backslashed characters and stores that into our format
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])#Then we use filter to match these words and substitute this with a '' to replace the character.
    #And return a list
    #print('Without special')
    return list(filtered_tokens)


def remove_stop_words(tokens):
    filtered_tokens=[word for word in tokens if word not in eng_stopwords]
    return list(filtered_tokens)

def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')#pattern Syntax-Match one character n times,match one character,match one single character,match character n times
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
    correct_tokens = [replace(word) for word in tokens]
    return list(correct_tokens)

'''Stopword removal and extra character removal'''
tokenizer=nltk.word_tokenize

#print(train_df['question2'])
#print(train_df.dtypes.index)
lemmatizer=nltk.stem.WordNetLemmatizer()

print('Processing question2(Training)')
train_df['question2']=train_df['question2'].apply(lambda x:str(x))
train_df['question2']=[word.lower() for word in train_df['question2']]
train_df['question2']=train_df['question2'].apply(tokenizer)

print('Processing question1(Training')
train_df['question1']=[word.lower() for word in train_df['question1']]
train_df['question1']=train_df['question1'].apply(tokenizer)

print('Processing question1(Testing)')
test_df['question1']=test_df['question1'].apply(lambda x:str(x))
test_df['question1']=[word.lower() for word in test_df['question1']]
test_df['question1']=test_df['question1'].apply(tokenizer)

print('Processing question2(Testing)')
test_df['question2']=test_df['question2'].apply(lambda x:str(x))
test_df['question2']=[word.lower() for word in test_df['question2']]
test_df['question2']=test_df['question2'].apply(tokenizer)

print('Lemmatizing Both')
for something in train_df['question1']:
    for item in something:
        item=lemmatizer.lemmatize(item)

for something in train_df['question2']:
    for item in something:
        item=lemmatizer.lemmatize(item)

for something in test_df['question1']:
    for item in something:
        item=lemmatizer.lemmatize(item)

for something in test_df['question2']:
    for item in something:
        item=lemmatizer.lemmatize(item)

print('Processing Question 2(Actually) Training')
train_df['question2']=train_df['question2'].apply(lambda x:remove_repeated_characters(x))
train_df['question2']=train_df['question2'].apply(lambda z:remove_special_characters_after_tokenization(z))
train_df['question2']=train_df['question2'].apply(lambda x:remove_stop_words(x))

print('Processing Question 1(Actually) Training')
train_df['question1']=train_df['question1'].apply(lambda x:remove_repeated_characters(x))
train_df['question1']=train_df['question1'].apply(lambda z:remove_special_characters_after_tokenization(z))
train_df['question1']=train_df['question1'].apply(lambda x:remove_stop_words(x))

print('Processing Question 2(Actually) Testing')
test_df['question2']=test_df['question2'].apply(lambda x:remove_repeated_characters(x))
test_df['question2']=test_df['question2'].apply(lambda z:remove_special_characters_after_tokenization(z))
test_df['question2']=test_df['question2'].apply(lambda x:remove_stop_words(x))

print('Processing Question 1(Actually) Testing')
test_df['question1']=test_df['question1'].apply(lambda x:remove_repeated_characters(x))
test_df['question1']=test_df['question1'].apply(lambda z:remove_special_characters_after_tokenization(z))
test_df['question1']=test_df['question1'].apply(lambda x:remove_stop_words(x))

'''Creating Embeddings[For both question together]'''

print('Creating Embeddings')
#embeddings=gensim.models.Word2Vec(train_df['question1']+train_df['question2'])

#print(embeddings)

tfidfvec = TfidfVectorizer(analyzer='word', ngram_range = (1,2), min_df = 150, max_features=500,lowercase=False,use_idf=True,tokenizer=tokenizer)

train_df['question1']=train_df['question1'].apply(lambda x:str(x))
train_df['question2']=train_df['question2'].apply(lambda x:str(x))

test_df['question1']=test_df['question1'].apply(lambda x:str(x))
test_df['question2']=test_df['question2'].apply(lambda x:str(x))

print('Combining Both Data')

train_df['Combined Data']=train_df['question1']+train_df['question2']
test_df['Combined Data']=test_df['question1']+test_df['question2']

target=train_df['is_duplicate']

tfidftrainVector=tfidfvec.fit_transform(train_df['Combined Data'])

tfidftestVector=tfidfvec.fit_transform(test_df['Combined Data'])

tfidftrainVector=tfidftrainVector.toarray()

model1=CatBoostClassifier(learning_rate=0.02,depth=8,iterations=500)

print('Fitting model')

model1.fit(tfidftrainVector,target)

predict=model1.predict(tfidftestVector.toarray())

test_df['predictions']=predict

test_df.to_csv('M:/PycharmProjects/AI+DL+CP/QQP/submission.csv',columns=['test_id','predictions'])

