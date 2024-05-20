# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Problem Statement 
#
# You need to build a model that is able to classify customer complaints based on the products/services. By doing so, you can segregate these tickets into their relevant categories and, therefore, help in the quick resolution of the issue.
#
# You will be doing topic modelling on the <b>.json</b> data provided by the company. Since this data is not labelled, you need to apply NMF to analyse patterns and classify tickets into the following five clusters based on their products/services:
#
# * Credit card / Prepaid card
#
# * Bank account services
#
# * Theft/Dispute reporting
#
# * Mortgages/loans
#
# * Others 
#
#
# With the help of topic modelling, you will be able to map each ticket onto its respective department/category. You can then use this data to train any supervised model such as logistic regression, decision tree or random forest. Using this trained model, you can classify any new customer complaint support ticket into its relevant department.

# %% [markdown]
# ## Pipelines that needs to be performed:
#
# You need to perform the following eight major tasks to complete the assignment:
#
# 1.  Data loading
#
# 2. Text preprocessing
#
# 3. Exploratory data analysis (EDA)
#
# 4. Feature extraction
#
# 5. Topic modelling 
#
# 6. Model building using supervised learning
#
# 7. Model training and evaluation
#
# 8. Model inference

# %% [markdown]
# ## Importing the necessary libraries

# %%
import tensorflow as tf
# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# %%
import json 
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
nlp = en_core_web_sm.load()
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint

import os


# %% [markdown]
# ## Loading the data
#
# The data is in JSON format and we need to convert it to a dataframe.

# %%
# Opening JSON file 
f = open('./complaints-2021-05-14_08_16.json') # Write the path to your data file and load it 

# returns JSON object as  
# a dictionary 
data = json.load(f)
df=pd.json_normalize(data)


# %% [markdown]
# ## Data preparation

# %%
# Inspect the dataframe to understand the given data.
display(df.head())
display(df.info())
display(df.describe(include='all'))

# %%
#print the column names
df.columns


# %%
#Assign new column names
df.columns = df.columns.str.lstrip('_')
df.columns = df.columns.str.replace('source.', '')
df.columns


# %%
# Assign nan in place of blanks in the complaints column
df[df['complaint_what_happened'] ==''] = np.nan


# %%
# df shape before dropna
print('df.shape before dropna =',df.shape)
#Remove all rows where complaints column is nan
df.dropna(subset='complaint_what_happened', inplace=True)
# df shape after dropna
print('df.shape after dropna =',df.shape)


# %% [markdown]
# ## Prepare the text for topic modeling
#
# Once you have removed all the blank complaints, you need to:
#
# * Make the text lowercase
# * Remove text in square brackets
# * Remove punctuation
# * Remove words containing numbers
#
#
# Once you have done these cleaning operations you need to perform the following:
# * Lemmatize the texts
# * Extract the POS tags of the lemmatized text and remove all the words which have tags other than NN[tag == "NN"].
#

# %% [markdown]
# #### Convert data types

# %%
df = df.convert_dtypes()
df.info()

# %% [markdown]
# #### Make the text lowercase

# %%
# Write your function here to clean the text and remove all the unnecessary elements.
df['clean_complaints'] = df['complaint_what_happened'].str.lower()


# %% [markdown]
# #### Create function to extract regex

# %%
def extract_regex(df, new_df, regexes):
    if not isinstance(regexes, list):
        regexes = [regexes]
    new_df = pd.DataFrame()
    for reg in regexes:
        if new_df.empty:
            new_df = pd.DataFrame(df[df['clean_complaints'].str.contains(reg, regex=True)]['clean_complaints'])            
        else:
            new_df = pd.concat([new_df, pd.DataFrame(df[df['clean_complaints'].str.contains(reg, regex=True)]['clean_complaints'])])
        
    # Apply the regular expressions to the DataFrame
    new_df['extracted'] = new_df['clean_complaints'].apply(lambda x: [match for regex in regexes for match in re.findall(regex, x)])
    # Display the 'extracted' column
    return pd.DataFrame((new_df['extracted']))


# %% [markdown]
# #### Remove text in square brackets

# %%
df_square_brackets = pd.DataFrame()
regex = r'(\[.*?\])'
extract_regex(df, df_square_brackets, regex)

# %%
df['clean_complaints'] = df['clean_complaints'].str.replace(regex, '', regex = True)

# %%
extract_regex(df, df_square_brackets, regex)

# %% [markdown]
# #### Remove punctuation

# %%
df_punctuation = pd.DataFrame()
regex = f'[{string.punctuation}]'
extract_regex(df, df_punctuation, regex)

# %%
# Remove punctuation from 'clean_complaints'
df['clean_complaints'] = df['clean_complaints'].str.replace(regex, '', regex=True)

# %%
extract_regex(df, df_punctuation, regex)

# %% [markdown]
# #### Remove words containing numbers

# %% [markdown]
# Filter words containing numbers anywhere

# %%
# Create a new DataFrame that contains rows that have square brackets and its text
df_word_num = pd.DataFrame()

# Define the regular expressions
regex = [r'\b[A-Za-z]+\d+\w*\b', r'\b\d+[A-Za-z]+\w*\b']

extract_regex(df, df_word_num, regex).head(30)

# %% [markdown]
# Filter words containing numbers in between

# %%
# Create a new DataFrame that contains rows that have square brackets and its text
df_word_num = pd.DataFrame()

# Define the regular expressions
regex = r'\b[A-Za-z]+\d+[A-Za-z]+\b'

extract_regex(df, df_word_num, regex).head(10)

# %% [markdown]
# Remove words containing numbers in between

# %%
# Remove punctuation from 'clean_complaints'
df['clean_complaints'] = df['clean_complaints'].str.replace(regex, '', regex=True)

# %%
extract_regex(df, df_word_num, regex).head(10)

# %% [markdown]
# #### Clean space

# %%
print(df['clean_complaints'][1])

# %%
# Removing leading/trailing whitespace and empty sentences
df['clean_complaints'] = df['clean_complaints'].apply(lambda x: '\n'.join(sent.strip() for sent in x.split('\n') if sent.strip() != ''))

# %%
print(df['clean_complaints'][1])

# %%
# Removing extra spaces between words.
df['clean_complaints'] = df['clean_complaints'].apply(lambda x: '\n'.join(' '.join(word.strip() for word in sent.split() if word.strip()!= '') for sent in x.split('\n') if sent.strip()!= ''))

# %%
print(df['clean_complaints'][1])

# %% [markdown]
# #### Drop empty rows

# %%
# df shape before dropna
print('df.shape before dropna =',df.shape)
#Remove all rows where complaints column is nan
df.dropna(subset='clean_complaints', inplace=True)
# Drop rows where column 'clean_complaints' is equal to ''
df = df[df['clean_complaints'] != '']
# df shape after dropna
print('df.shape after dropna =',df.shape)
# reset index
df.reset_index(drop=True, inplace=True)

# %% [markdown]
# #### Drop duplicates

# %%
# df shape before drop_duplicates
print('df.shape before drop_duplicates =',df.shape)
# Drop duplicate rows based on column 'clean_complaints'
df = df.drop_duplicates(subset='clean_complaints')
# df shape after drop_duplicates
print('df.shape after drop_duplicates =',df.shape)

# %%
if os.path.isfile('df.csv'):
  # load df_clean
  df = pd.read_csv('df.csv')
  if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)
else:
    df.to_csv('df.csv', index=False)


# %% [markdown]
# #### Lemmatize the texts

# %%
#Write your function to Lemmatize the texts
def lemmatize(sent):        
    spacy.prefer_gpu()
    doc = nlp(sent)
    return ' '.join([token.lemma_ for token in doc])


# %%
if os.path.isfile('df_clean.csv'):
    # load df_clean
    df_clean = pd.read_csv('df_clean.csv')
    if 'Unnamed: 0' in df_clean.columns:
      df_clean.drop('Unnamed: 0', axis=1, inplace=True)
else:
    #tag remote collab
    df = pd.read_csv('df.csv')

    #Create a dataframe('df_clean') that will have only the complaints and the lemmatized complaints 
    df_clean = pd.DataFrame()

    # initialize 'complaints' column
    df_clean['complaints'] = df['clean_complaints']

    # process 'complaints_lemmatized' column
    df_clean['complaints_lemmatized'] = pd.DataFrame(df_clean['complaints'].apply(lambda x: '\n'.join(lemmatize(sent) for sent in x.split('\n'))))

    # Store df_clean for later use
    df_clean.to_csv('df_clean.csv', index=False)

# %%
print(df_clean['complaints_lemmatized'][0])

# %%
df_clean


# %%
def pos_tag(text):
    # write your code here
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    #this column should contain lemmatized text with all the words removed which have tags other than NN[tag == "NN"].
    return lemmatize(' '.join([word for word, tag in pos_tags if tag.startswith('NN')]))


# %%
#Write your function to extract the POS tags 

if os.path.isfile('df_clean_v1.csv'):
  #tag remote collab
  df = pd.read_csv('df.csv')
  # load df_clean
  df_clean = pd.read_csv('df_clean_v1.csv')
  if 'Unnamed: 0' in df_clean.columns:
    df_clean.drop('Unnamed: 0', axis=1, inplace=True)
else:
  #tag remote collab
  df = pd.read_csv('df.csv')
  #tag remote collab
  df_clean = pd.read_csv('df_clean.csv')

  nltk.download('punkt')
  # Make sure you have the necessary NLTK data downloaded
  nltk.download('averaged_perceptron_tagger')

  df_clean['complaint_POS_removed'] = pd.DataFrame(df_clean['complaints'].apply(lambda x: '\n'.join(pos_tag(sent) for sent in x.split('\n'))))
  # Store df_clean for later use
  df_clean.to_csv('df_clean_v1.csv', index=False)


# %%
#The clean dataframe should now contain the raw complaint, lemmatized complaint and the complaint after removing POS tags.
df_clean

# %% [markdown]
# ## The personal details of customer has been masked in the dataset with xxxx. Let's remove the masked text as this will be of no use for our analysis

# %%
df_clean['complaint_POS_removed'] = df_clean['complaint_POS_removed'].str.replace(r'xxxx*','', regex =True)

# %%
# Replace NaN values with an empty string
df_clean['complaint_POS_removed'] = df_clean['complaint_POS_removed'].fillna('')
# Removing leading/trailing whitespace and empty sentences
df_clean['complaint_POS_removed'] = df_clean['complaint_POS_removed'].apply(lambda x: '\n'.join(sent.strip() for sent in x.split('\n') if sent.strip() != ''))
# Removing extra spaces between words.
df_clean['complaint_POS_removed'] = df_clean['complaint_POS_removed'].apply(lambda x: '\n'.join(' '.join(word.strip() for word in sent.split() if word.strip()!= '') for sent in x.split('\n') if sent.strip()!= ''))

# %%
#All masked texts has been removed
df_clean

# %% [markdown]
# ## Exploratory data analysis to get familiar with the data.
#
# Write the code in this task to perform the following:
#
# *   Visualise the data according to the 'Complaint' character length
# *   Using a word cloud find the top 40 words by frequency among all the articles after processing the text
# *   Find the top unigrams,bigrams and trigrams by frequency among all the complaints after processing the text. ‘
#
#
#

# %%
# Write your code here to visualise the data according to the 'Complaint' character length

# Create a new column 'complaint_length' that contains the length of each complaint
complaint_length = df_clean['complaints_lemmatized'].apply(len)

# Set the figure size
plt.figure(figsize=(13, 5))

# Plot a histogram of the complaint lengths
sns.histplot(complaint_length, edgecolor='white', bins=50, alpha=0.55, kde=True)
plt.xlabel('Complaint Length')
plt.ylabel('Frequency')
plt.title('Distribution of Complaint Lengths')
plt.show()

# %% [markdown]
# Distribution of complaint lengths is strongly right-skewed, indicating that most complaints are short, but there are a few very long ones that extend the range significantly.

# %% [markdown]
# #### Find the top 40 words by frequency among all the articles after processing the text.

# %%
# Get the list of English stop words
stop_words = nlp.Defaults.stop_words

# Create a new column with stop words removed
df_clean['complaint_POS_removed'] = df_clean['complaint_POS_removed'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))


# %%
#Using a word cloud find the top 40 words by frequency among all the articles after processing the text

from wordcloud import WordCloud
# Combine all the complaints into a single string
all_complaints = ' '.join(df_clean['complaint_POS_removed'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, max_words=40).generate(all_complaints)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# %%
#Removing -PRON- from the text corpus
# df_clean['Complaint_clean'] = df_clean['complaint_POS_removed'].str.replace('-PRON-', '')
if os.path.isfile('df_clean_v2.csv'):
  #tag remote collab
  df = pd.read_csv('df.csv')
  # load df_clean
  df_clean = pd.read_csv('df_clean_v2.csv')
  if 'Unnamed: 0' in df_clean.columns:
    df_clean.drop('Unnamed: 0', axis=1, inplace=True)
else:
    #tag remote collab
    df = pd.read_csv('df.csv')
    # Define a function to replace a token
    def remove_PRON(sent):
        spacy.prefer_gpu()
        doc = nlp(sent)
        return ' '.join([token.text for token in doc if token.pos_ !='PRON'])

    # Apply the function to the 'complaint_POS_removed' column
    df_clean['Complaint_clean'] = df_clean['complaint_POS_removed'].apply(remove_PRON)

    df_clean.to_csv('df_clean_v2.csv', index=False)


# %% [markdown]
# #### Find the top unigrams,bigrams and trigrams by frequency among all the complaints after processing the text.

# %% [markdown]
# - __Unigram__ means taking only one word at a time.
# - __Bigram__ means taking two words at a time.
# - __Trigram__ means taking three words at a time. 
#
# Source: `https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/`

# %%
#Write your code here to find the top 30 unigram frequency among the complaints in the cleaned datafram(df_clean). 
import collections

# Assuming 'complaint_POS_removed_clean' is the column in df_clean that contains the cleaned complaints
complaints = df_clean['complaint_POS_removed_clean'].str.split().sum()

# Count the frequency of each word
word_counts = collections.Counter(complaints)

# Get the top 30 most common words
top_30_words = word_counts.most_common(30)

print(top_30_words)



# %%
#Print the top 10 words in the unigram frequency



# %%
#Write your code here to find the top 30 bigram frequency among the complaints in the cleaned datafram(df_clean). 


# %%
#Print the top 10 words in the bigram frequency

# %%
#Write your code here to find the top 30 trigram frequency among the complaints in the cleaned datafram(df_clean). 


# %%
#Print the top 10 words in the trigram frequency

# %% [markdown]
# ## Feature Extraction
# Convert the raw texts to a matrix of TF-IDF features
#
# **max_df** is used for removing terms that appear too frequently, also known as "corpus-specific stop words"
# max_df = 0.95 means "ignore terms that appear in more than 95% of the complaints"
#
# **min_df** is used for removing terms that appear too infrequently
# min_df = 2 means "ignore terms that appear in less than 2 complaints"

# %%
#Write your code here to initialise the TfidfVectorizer 



# %% [markdown]
# #### Create a document term matrix using fit_transform
#
# The contents of a document term matrix are tuples of (complaint_id,token_id) tf-idf score:
# The tuples that are not there have a tf-idf score of 0

# %%
#Write your code here to create the Document Term Matrix by transforming the complaints column present in df_clean.


# %% [markdown]
# ## Topic Modelling using NMF
#
# Non-Negative Matrix Factorization (NMF) is an unsupervised technique so there are no labeling of topics that the model will be trained on. The way it works is that, NMF decomposes (or factorizes) high-dimensional vectors into a lower-dimensional representation. These lower-dimensional vectors are non-negative which also means their coefficients are non-negative.
#
# In this task you have to perform the following:
#
# * Find the best number of clusters 
# * Apply the best number to create word clusters
# * Inspect & validate the correction of each cluster wrt the complaints 
# * Correct the labels if needed 
# * Map the clusters to topics/cluster names

# %%
from sklearn.decomposition import NMF

# %% [markdown]
# ## Manual Topic Modeling
# You need to do take the trial & error approach to find the best num of topics for your NMF model.
#
# The only parameter that is required is the number of components i.e. the number of topics we want. This is the most crucial step in the whole topic modeling process and will greatly affect how good your final topics are.

# %%
#Load your nmf_model with the n_components i.e 5
num_topics = #write the value you want to test out

#keep the random_state =40
nmf_model = #write your code here

# %%
nmf_model.fit(dtm)
len(tfidf.get_feature_names())

# %%
#Print the Top15 words for each of the topics


# %%
#Create the best topic for each complaint in terms of integer value 0,1,2,3 & 4



# %%
#Assign the best topic to each of the cmplaints in Topic Column

df_clean['Topic'] = #write your code to assign topics to each rows.

# %%
df_clean.head()

# %%
#Print the first 5 Complaint for each of the Topics
df_clean=df_clean.groupby('Topic').head(5)
df_clean.sort_values('Topic')

# %% [markdown]
# #### After evaluating the mapping, if the topics assigned are correct then assign these names to the relevant topic:
# * Bank Account services
# * Credit card or prepaid card
# * Theft/Dispute Reporting
# * Mortgage/Loan
# * Others

# %%
#Create the dictionary of Topic names and Topics

Topic_names = {   }
#Replace Topics with Topic Names
df_clean['Topic'] = df_clean['Topic'].map(Topic_names)

# %%
df_clean

# %% [markdown]
# ## Supervised model to predict any new complaints to the relevant Topics.
#
# You have now build the model to create the topics for each complaints.Now in the below section you will use them to classify any new complaints.
#
# Since you will be using supervised learning technique we have to convert the topic names to numbers(numpy arrays only understand numbers)

# %%
#Create the dictionary again of Topic names and Topics

Topic_names = {   }
#Replace Topics with Topic Names
df_clean['Topic'] = df_clean['Topic'].map(Topic_names)

# %%
df_clean

# %%
#Keep the columns"complaint_what_happened" & "Topic" only in the new dataframe --> training_data
training_data=

# %%
training_data

# %% [markdown]
# ####Apply the supervised models on the training data created. In this process, you have to do the following:
# * Create the vector counts using Count Vectoriser
# * Transform the word vecotr to tf-idf
# * Create the train & test data using the train_test_split on the tf-idf & topics
#

# %%

#Write your code to get the Vector count


#Write your code here to transform the word vector to tf-idf

# %% [markdown]
# You have to try atleast 3 models on the train & test data from these options:
# * Logistic regression
# * Decision Tree
# * Random Forest
# * Naive Bayes (optional)
#
# **Using the required evaluation metrics judge the tried models and select the ones performing the best**

# %%
# Write your code here to build any 3 models and evaluate them using the required metrics





# %%



