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

# %%
# df = df.convert_dtypes()
# df.info()

# %% [markdown]
# #### Make the text lowercase

# %%
# Write your function here to clean the text and remove all the unnecessary elements.
df['clean_complaints'] = df['complaint_what_happened'].str.lower()


# %% [markdown]
# #### Create function to extract regex

# %%
def extract_regex(df, new_df, regex):
    # Create a new DataFrame that contains rows that have square brackets and its text
    new_df = pd.DataFrame(df[df['clean_complaints'].str.contains(regex, regex = True)]['clean_complaints'])

    # Extract the square brackets and its text from the 'clean_complaints' column and store it in a new column 'extracted'
    new_df['extracted'] = new_df['clean_complaints'].apply(lambda x: re.findall(regex, x))

    # Display the 'extracted' column
    display(pd.DataFrame((new_df['extracted'])))


# %% [markdown]
# #### Remove text in square brackets

# %%
df_square_brackets = pd.DataFrame()
regex = r'(\[.*?\])'
extract_regex(df, df_square_brackets, regex)

# %%
df['clean_complaints'] = df['clean_complaints'].str.replace(r'(\[.*?\])', '', regex = True)

# %%
df_square_brackets = pd.DataFrame()
regex = r'(\[.*?\])'
extract_regex(df, df_square_brackets, regex)

# %% [markdown]
# #### Remove punctuation

# %%
df_punctuation = pd.DataFrame()
regex = f'[{string.punctuation}]'
extract_regex(df, df_punctuation, regex)

# %%
# Remove punctuation from 'clean_complaints'
df['clean_complaints'] = df['clean_complaints'].str.replace(f'[{string.punctuation}]', '', regex=True)

# %%
df_punctuation = pd.DataFrame()
regex = f'[{string.punctuation}]'
extract_regex(df, df_punctuation, regex)

# %% [markdown]
# #### Remove words containing numbers

# %% [markdown]
# Filter words containing numbers anywhere

# %%
# Create a new DataFrame that contains rows that have square brackets and its text
df_word_num =pd.DataFrame(df[df['clean_complaints'].str.contains(r'\b[A-Za-z]+\d+\w*\b', regex=True) | df['clean_complaints'].str.contains(r'\b\d+[A-Za-z]+\w*\b', regex=True)]['clean_complaints'])

# Define the regular expressions
regexes = [r'\b[A-Za-z]+\d+\w*\b', r'\b\d+[A-Za-z]+\w*\b']

# Apply the regular expressions to the DataFrame
df_word_num['extracted'] = df_word_num['clean_complaints'].apply(lambda x: [match for regex in regexes for match in re.findall(regex, x)])

# Display the 'extracted' column
display(pd.DataFrame(df_word_num['extracted']).head(30))

# %% [markdown]
# Filter words containing numbers in between

# %%
# Create a new DataFrame that contains rows that have square brackets and its text
df_word_num =pd.DataFrame(df[df['clean_complaints'].str.contains(r'\b[A-Za-z]+\d+[A-Za-z]+\b', regex=True)])

# Define the regular expressions
regexes = [r'\b[A-Za-z]+\d+[A-Za-z]+\b']

# Apply the regular expressions to the DataFrame
df_word_num['extracted'] = df_word_num['clean_complaints'].apply(lambda x: [match for regex in regexes for match in re.findall(regex, x)])


# Display the 'extracted' column
display(pd.DataFrame(df_word_num['extracted']).head(10))

# %% [markdown]
# Remove words containing numbers in between

# %%
# Remove punctuation from 'clean_complaints'
df['clean_complaints'] = df['clean_complaints'].str.replace(r'\b[A-Za-z]+\d+[A-Za-z]+\b', '', regex=True)

# %%
# Create a new DataFrame that contains rows that have square brackets and its text
df_word_num =pd.DataFrame(df[df['clean_complaints'].str.contains(r'\b[A-Za-z]+\d+[A-Za-z]+\b', regex=True)])

# Define the regular expressions
regexes = [r'\b[A-Za-z]+\d+[A-Za-z]+\b']

# Apply the regular expressions to the DataFrame
df_word_num['extracted'] = df_word_num['clean_complaints'].apply(lambda x: [match for regex in regexes for match in re.findall(regex, x)])


# Display the 'extracted' column
display(pd.DataFrame(df_word_num['extracted']).head(10))

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
# #### Lemmatize the texts

# %%
#Write your function to Lemmatize the texts
def lemmatize(sent):
    doc = nlp(sent)
    return ' '.join([token.lemma_ for token in doc])


# %%
#Create a dataframe('df_clean') that will have only the complaints and the lemmatized complaints 
df_clean = pd.DataFrame()

# initialize 'complaints' column
df_clean['complaints'] = df['clean_complaints']

# process 'complaints_lemmatized' column
df_clean['complaints_lemmatized'] = pd.DataFrame(df_clean['complaints'].apply(lambda x: '\n'.join(lemmatize(sent) for sent in x.split('\n'))))


# %%
print(df_clean['complaints_lemmatized'][1])

# %%
df_clean

# %%
#Write your function to extract the POS tags 

def pos_tag(text):
  # write your code here



df_clean["complaint_POS_removed"] =  #this column should contain lemmatized text with all the words removed which have tags other than NN[tag == "NN"].


# %%
#The clean dataframe should now contain the raw complaint, lemmatized complaint and the complaint after removing POS tags.
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

# %% [markdown]
# #### Find the top 40 words by frequency among all the articles after processing the text.

# %%
#Using a word cloud find the top 40 words by frequency among all the articles after processing the text


# %%
#Removing -PRON- from the text corpus
df_clean['Complaint_clean'] = df_clean['complaint_POS_removed'].str.replace('-PRON-', '')

# %% [markdown]
# #### Find the top unigrams,bigrams and trigrams by frequency among all the complaints after processing the text.

# %%
#Write your code here to find the top 30 unigram frequency among the complaints in the cleaned datafram(df_clean). 


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
# ## The personal details of customer has been masked in the dataset with xxxx. Let's remove the masked text as this will be of no use for our analysis

# %%
df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.replace('xxxx','')

# %%
#All masked texts has been removed
df_clean

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



