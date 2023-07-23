# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:37:35 2023

@author: julius
"""


# coding: utf-8

# In[1]:

from __future__ import division
import pandas as pd
import math as math
import random
import numpy as np
import collections
import csv
import os
from surprise.model_selection import RandomizedSearchCV, cross_validate
from surprise import SVD, Reader, accuracy, Dataset
from itertools import product


# LOADING AND TRANSFORMING DATASET

# In[2]:
    
# set wd
os.chdir("...")

# load dataset
data_pd = pd.read_csv("useritemmatrix.csv")

# rename columns to fit surprise package
data_pd.rename(columns={'userId': 'user_id', 'itemId': 'item_id', 'interaction': 'raw_ratings'}, inplace=True)

# set wd
os.chdir("...")


# # HYPERPARAMETER TUNING

# In[3]:
 
# make data fit for surprise
data = Dataset.load_from_df(data_pd[['user_id', 'item_id', 'raw_ratings']],
                            reader = Reader(rating_scale=(0, 1)))   

# parameter possibilities
param_distributions = {'n_factors': [100, 200, 500], 
                'n_epochs': [100],
                'lr_bu': [1e-01, 1e-02, 1e-03],
                'lr_bi': [1e-01, 1e-02, 1e-03],
                'lr_pu': [1e-01, 1e-02, 1e-03],
                'lr_qi': [1e-01, 1e-02, 1e-03],
                'reg_bu': [1e-05, 1e-06, 1e-07, 1e-08],
                'reg_bi': [1e-05, 1e-06, 1e-07, 1e-08],
                'reg_pu': [1e-05, 1e-06, 1e-07, 1e-08],
                'reg_qi': [1e-05, 1e-06, 1e-07, 1e-08]
}

# run randomized grid search
rand_search = RandomizedSearchCV(SVD,
                    param_distributions, 
                    n_iter=100, 
                    measures=['rmse'], 
                    cv=5, 
                    refit=False, 
                    return_train_measures=False, 
                    n_jobs=-2, 
                    pre_dispatch='2*n_jobs', 
                    random_state=12, 
                    joblib_verbose=2)

# fit grid search
rand_search.fit(data)

# save results in CSV
cv_results_rand = pd.DataFrame(rand_search.cv_results)
cv_results_rand.to_csv('cv_results_rand.csv', encoding='utf-8')

# optimal hyperparameters
# num_factors
factors = 200
# regularization bias terms
reg_b = 1e-07
# regularization latent factors
reg_q = 1e-06


# RANDOMLY SELECTING COLD USERS

# In[4]:

# checking users who have many purchases
user_freq_df = pd.DataFrame.from_dict(collections.Counter(data_pd['user_id']),orient='index').reset_index()
user_freq_df = user_freq_df.rename(columns={'index':'user_id', 0:'freq'})

# percentage of total number of users to set as cold user
perc_cold_users = 0.25
nr_of_cold_users = int(math.floor(len(user_freq_df)*perc_cold_users))
# select the [nr_of_cold_users] with the highest number of interactions
# cold_users = user_freq_df.sort_values(by='freq',ascending=False).head(nr_of_cold_users)
cold_users = user_freq_df.sample(nr_of_cold_users,random_state=1)
#cold_users = cold_users.get_value(index=range(0,(nr_of_cold_users)),col=0,takeable=True)
cold_users = cold_users.iloc[range(0, nr_of_cold_users), 0]

print('Selecting ' + str(nr_of_cold_users) + ' cold user(s)')

# SETTINGS FOR SHOWN ITEMS (ranking lengths and item frequency threshold) AND COMPUTING THE GINI, ENTROPY AND POPENT SCORES FOR THE ITEMS

# In[7]:

# compute purchase purchase/return frequency per item
item_freq_counter = collections.Counter(data_pd['item_id'])
item_freq_df = pd.DataFrame.from_dict(item_freq_counter,orient='index').reset_index()
item_freq_df = item_freq_df.rename(columns={'index':'item_id', 0:'freq'})

# produce list of items which are at least interacted with [threshold_item] times
threshold_item = 10
threshold_item_df = item_freq_df[item_freq_df['freq']>=threshold_item]['item_id']
threshold_freq_df = item_freq_df[item_freq_df['freq']>=threshold_item]

# ### ENTROPY0 SCORE ###

# Randomly select 5% of the user IDs and 5% of item IDs
num_user_ids = int(0.05 * len(data_pd['user_id'].unique()))
num_item_ids = int(0.05 * len(threshold_item_df))
selected_user_ids = pd.Series(data_pd['user_id'].unique()).sample(n=num_user_ids, random_state=42)
selected_item_ids = pd.Series(threshold_item_df).sample(n=num_item_ids, random_state=42)

# Generate all combinations of selected user IDs and item IDs
combinations = list(product(selected_user_ids, selected_item_ids))

# Create a new DataFrame with all combinations
data_entropy0 = pd.DataFrame(combinations, columns=['user_id', 'item_id'])

# Merge with the original DataFrame to retain existing raw_ratings values
data_entropy0 = data_entropy0.merge(data_pd[data_pd['item_id'].isin(threshold_item_df)], on=['user_id', 'item_id'], how='left')

# Fill missing raw_ratings with 0
data_entropy0['raw_ratings'].fillna(0, inplace=True)

# drop index
data_entropy0.reset_index(drop=True, inplace=True)

# Count the occurrences of 1's and 0's in raw_ratings
rating_counts = data_entropy0['raw_ratings'].value_counts()

# Access the counts
print("Count of 1's:", rating_counts[1])
print("Count of 0's:", rating_counts[0])

# function to compute entropy
def entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    for iterator in probs:
        ent -= iterator * np.log2(iterator)

    return ent

unique_item_id = selected_item_ids
entropy_list = np.empty(shape=(len(unique_item_id), 2), dtype=object)
j = 0

# loop over all itemId's and compute the entropy for each item
for i in unique_item_id:
    if j < 2253:
        j += 1
        continue
    item_i_df = data_entropy0[data_entropy0['item_id'] == i]
    entropy_list[j] = [i, entropy(item_i_df['raw_ratings'])]
    j += 1

# transform to pandas DataFrame
to_df = {'item_id' : entropy_list[:,0],'entropy0' : entropy_list[:,1]}
ent_items_df = pd.DataFrame(to_df)
ent_items_df.sort_values(by='entropy0',inplace=True,ascending=False)

print('Computed entropy0 scores for all items')

# to CSV
ent_items_df.to_csv('entropy0.csv', encoding='utf-8')
ent_items_df = pd.read_csv('entropy0.csv', encoding='utf-8')

### GREEDY EXTENT STRATEGY ###

j = 0
greedy_list = np.empty(shape=(len(threshold_item_df), 2), dtype=object)

for ids in threshold_item_df:
    data_greedy = data_pd[data_pd['item_id'] == ids]

    model = SVD(n_factors=factors, 
                    n_epochs=100,
                    biased=True,
                    reg_all=None,
                    lr_bu=None,
                    lr_bi=None,
                    lr_pu=None,
                    lr_qi=None,
                    reg_bu=reg_b,
                    reg_bi=reg_b,
                    reg_pu=reg_q,
                    reg_qi=reg_q,
                    random_state=None,
                    verbose=False)

    data = Dataset.load_from_df(data_greedy[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))

    train = data.build_full_trainset()

    # Compute the RMSE with cross-validation
    cv_results = cross_validate(model, data, measures=['RMSE'], cv=5, verbose=False)
    rmse = np.mean(cv_results['test_rmse'])
    
    # item_i_df = data_entropy0[data_entropy0['item_id'] == i]
    greedy_list[j, 0] = ids
    greedy_list[j, 1] = rmse
    j += 1

# transform to pandas DataFrame
to_df = {'item_id' : greedy_list[:,0],'rmse' : greedy_list[:,1]}
greedy_items_df = pd.DataFrame(to_df)
greedy_items_df.sort_values(by='rmse',inplace=True,ascending=True)

print('Computed greedy extent scores for all items')

# to csv
greedy_items_df.to_csv('greedy.csv', encoding='utf-8')
greedy_items_df = pd.read_csv('greedy.csv', encoding='utf-8')


### ENTROPY0 POPULARITY STRATEGY ###
# prepare item entropies for merging
item_freq_entropy0 = item_freq_df[item_freq_df['item_id'].isin(ent_items_df['item_id'])]
# merge frequencies and entropies
entpop_items_df = pd.merge(item_freq_entropy0, ent_items_df, on='item_id')

print('Preparation Entropy0 popularity for all items completed')


### GREEDY EXTENT POPULARITY STRATEGY ###
# prepare item greedy for merging
# merge frequencies and greedy extent
greedypop_items_df = pd.merge(threshold_freq_df, greedy_items_df, on='item_id')

print('Preparation greedy extent scores for all items completed')


# WEIGHT OPTIMIZATION MULTIPLE HEURISTIC STRATAGIES

# In[5]:

# model
model = SVD(n_factors=factors, 
                        n_epochs=100,
                        biased=True,
                        reg_all=None,
                        lr_bu=None,
                        lr_bi=None,
                        lr_pu=None,
                        lr_qi=None,
                        reg_bu=reg_b,
                        reg_bi=reg_b,
                        reg_pu=reg_q,
                        reg_qi=reg_q,
                        random_state=None,
                        verbose=False)    

#entpop
filename = 'weights_entpop.csv'
csvfile = open(filename, 'w+')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(['Ranking strategy','Nr. of shown items','Nr. of cold users','RMSE','Weight popularity','Weight entropy0'])

weight_pop_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
weight_ent_list = [0.2, 0.4, 0.6, 0.8, 1]
nr_of_shown_items_list = [10, 25, 50, 100]

for k in weight_pop_list:
    for l in weight_ent_list: 
        # set weights for the popent score
        weight_popularity = k
        weight_entropy0 = l
        # compute popent score
        entpop_items_df['entpop'] = weight_popularity*np.log10(entpop_items_df['freq'])+weight_entropy0*entpop_items_df['entropy0']
        entpop_items_df.sort_values(by='entpop',inplace=True,ascending=False)

        print('Computed entpop scores for all items')
        print(k) 
        print(l)

        for m in nr_of_shown_items_list:
            # set the number of items to show to the cold user
            nr_of_shown_items = m
            print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))
            number_of_shown_items = str(nr_of_shown_items)
            # ENTPOP STRATEGY
            # select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
            entpop_items = entpop_items_df.head(nr_of_shown_items)
            entpop_items = np.array(entpop_items['item_id'])
            print('Computed ranking using entropy0 popularity strategy')
            print('Computing results')
            ranking_strategy = 'Entropy0 popularity strategy'
            # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
            train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(entpop_items))]
            test_pd = data_pd[(data_pd.user_id.isin(cold_users)) & (~data_pd.item_id.isin(entpop_items))]
            train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                            reader = Reader(rating_scale=(0, 1)))
            test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
            # Retrieve the trainset.
            train = train.build_full_trainset()
            # fit model
            model.fit(train)
            # predict
            pred = model.test(test)
            # compute RMSE
            rmse = accuracy.rmse(pred)
            print('RMSE computed for entropy0 popularity strategy')
            # store result in csv file
            writer.writerow([ranking_strategy, number_of_shown_items, 
                             nr_of_cold_users, rmse, k, l])

csvfile.close()

#greedypop
filename = 'weights_greedypop.csv'
csvfile = open(filename, 'w+')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(['Ranking strategy','Nr. of shown items','Nr. of cold users','RMSE','Weight popularity','Weight greedy extent'])

weight_pop_list = [0.001, 0.01, 0.2, 0.4, 0.6, 0.8, 1]
weight_greedy_list = [0.2, 0.4, 0.6, 0.8, 1]
nr_of_shown_items_list = [10, 25, 50, 100]

for k in weight_pop_list:
    for l in weight_greedy_list: 
        # set weights for the greedypop score
        weight_popularity = k
        weight_greedy = l
        # compute greedypop score
        greedypop_items_df['greedypop'] = weight_popularity*np.log10(greedypop_items_df['freq'])+weight_greedy*-1*greedypop_items_df['rmse']
        greedypop_items_df.sort_values(by='greedypop',inplace=True,ascending=False)

        print('Computed greedypop scores for all items')
        print(k) 
        print(l)

        for m in nr_of_shown_items_list:
            # set the number of items to show to the cold user
            nr_of_shown_items = m
            print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))
            number_of_shown_items = str(nr_of_shown_items)
            # greedypop STRATEGY
            # select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
            greedypop_items = greedypop_items_df.head(nr_of_shown_items)
            greedypop_items = np.array(greedypop_items['item_id'])
            print('Computed ranking using greedy popularity strategy')

            print('Computing results')
            ranking_strategy = 'Greedy extent popularity strategy'
            # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
            train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(greedypop_items))]
            # cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
            cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(greedypop_items))]['user_id'])
            test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(greedypop_items))]
            
            train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                        reader = Reader(rating_scale=(0, 1)))
            
            test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
            
            # Retrieve the trainset.
            train = train.build_full_trainset()
            
            # fit model
            model.fit(train)
            
            # predict
            pred = model.test(test)
            
            # compute RMSE
            rmse = accuracy.rmse(pred)
            
            print('RMSE computed for greedy extent poplarity strategy')
            # store result in csv file
            writer.writerow([ranking_strategy, number_of_shown_items, 
                             nr_of_cold_users, rmse, k, l])

csvfile.close()



# COMPOSING THE RESULTS

# In[6]:

# number of items to show to the cold user
item_to_be_shown = [10, 25, 50, 100]

# run loop for all strategies and set size of shown items
for nr_of_shown_items in item_to_be_shown:
    print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))
    
    # SELECTING ITEMS FOR EACH STRATEGY
    
    ### RANDOM STRATEGY ###
    # select [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) at random
    #random_items = random.sample(threshold_item_df,nr_of_shown_items)
    random_items = random.sample(sorted(threshold_item_df), nr_of_shown_items)
    random_items = np.array(random_items)
    print('Computed ranking using random strategy')
    
    ### ENTROPY0 STRATEGY ###
    # select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest entropy
    ent_items = ent_items_df.head(nr_of_shown_items)['item_id']
    ent_items = np.array(ent_items)
    print('Computed ranking using entropy0 strategy')
    
    ### GREEDY EXTENT STRATEGY ###
    greedy_items = greedy_items_df.head(nr_of_shown_items)['item_id']
    greedy_items = np.array(greedy_items)
    print('Computed ranking using greedy extent strategy')
    
    ### RANDOM POPULARITY STRATEGY ###
    # first select % of popular items
    pop = 0.1
    
    # sample popular items
    randpop_pop = item_freq_counter.most_common(int(pop * nr_of_shown_items))
    randpop_pop = [x[0] for x in randpop_pop]
    randpop_pop = np.array(randpop_pop)
    
    # Create a mask to filter out the items that are already in popular items
    mask = np.isin(threshold_item_df, randpop_pop, invert=True)
    
    # Apply the mask to the dataset to get the remaining items
    remaining_items = threshold_item_df[mask]
    
    # randomly sample remaining 1-pop percentage
    randpop_rand = random.sample(sorted(remaining_items), int((1-pop) * nr_of_shown_items))
    randpop_rand = np.array(randpop_rand)
    
    # Concatenate popular and random sample
    randpop_items = np.concatenate((randpop_pop, randpop_rand))
    
    print('Computed ranking using random popularity strategy')
        
    
    # COMPUTING THE RESULTS FOR EACH RANKING STRATEGY
    
    # number of shown items
    number_of_shown_items = str(nr_of_shown_items)
    
    print('Computing results')
    
    # model
    model = SVD(n_factors=factors, 
                        n_epochs=100,
                        biased=True,
                        reg_all=None,
                        lr_bu=None,
                        lr_bi=None,
                        lr_pu=None,
                        lr_qi=None,
                        reg_bu=reg_b,
                        reg_bi=reg_b,
                        reg_pu=reg_q,
                        reg_qi=reg_q,
                        random_state=None,
                        verbose=False)
    
    ### RANDOM STRATEGY ###
    ranking_strategy = 'Random strategy'
    # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(random_items))]
    # cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(random_items))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(random_items))]
    
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    
    # Retrieve the trainset.
    train = train.build_full_trainset()
    
    # fit model
    model.fit(train)
    
    # predict
    pred = model.test(test)
    
    # compute RMSE
    rmse = accuracy.rmse(pred)
    
    print('RMSE computed for random strategy')
    
    # store result in csv file
    filename = str('final_results2_shown_items_' + number_of_shown_items + '.csv')
    csvfile = open(filename, 'w+')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Ranking strategy','Nr. of shown items','Nr. of cold users','RMSE'])
    writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])
    
    ### ENTROPY0 STRATEGY ###
    ranking_strategy = 'Entropy0 strategy'
    # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(ent_items))]
    test_pd = data_pd[(data_pd.user_id.isin(cold_users)) & (~data_pd.item_id.isin(ent_items))]
    
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    
    # Retrieve the trainset.
    train = train.build_full_trainset()
    
    # fit model
    model.fit(train)
    
    # predict
    pred = model.test(test)
    
    # compute RMSE
    rmse = accuracy.rmse(pred)
    
    print('RMSE computed for entropy0 strategy')
    
    # store result in csv file
    writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])
    
    ### GREEDY EXTENT STRATEGY ###
    ranking_strategy = 'Greedy extent strategy'
    # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(greedy_items))]
    # cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(greedy_items))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(greedy_items))]
    
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    
    # Retrieve the trainset.
    train = train.build_full_trainset()
    
    # fit model
    model.fit(train)
    
    # predict
    pred = model.test(test)
    
    # compute RMSE
    rmse = accuracy.rmse(pred)
    
    print('RMSE computed for greedy extent strategy')
    
    # store result in csv file
    writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])
    
    ### RANDOM POPULARITY STRATEGY ###
    ranking_strategy = 'Random popularity strategy'
    # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(randpop_items))]
    # cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(randpop_items))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(randpop_items))]
    
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    
    # Retrieve the trainset.
    train = train.build_full_trainset()
    
    # fit model
    model.fit(train)
    
    # predict
    pred = model.test(test)
    
    # compute RMSE
    rmse = accuracy.rmse(pred)
    
    print('RMSE computed for random popularity strategy')
    
    # store result in csv file
    writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse])
    csvfile.close()


# SENSITIVITY ANALYSIS ENTROPY0
    
# In[7]:

# function to compute entropy
def entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    for iterator in probs:
        ent -= iterator * np.log2(iterator)

    return ent

# store result in csv file
filename = str('sensitivity entropy0.csv')
csvfile = open(filename, 'w+')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(['Ranking strategy','Nr. of shown items','Nr. of cold users','RMSE', 'Sample'])

for k in range(5):
    print(k)
    
    ### ENTROPY0 SCORE ###
    
    # Randomly select 5% of the user IDs and 5% of item IDs
    num_user_ids = int(0.05 * len(data_pd['user_id'].unique()))
    num_item_ids = int(0.05 * len(threshold_item_df))
    selected_user_ids = pd.Series(data_pd['user_id'].unique()).sample(n=num_user_ids)
    selected_item_ids = pd.Series(threshold_item_df).sample(n=num_item_ids)
    
    # Generate all combinations of selected user IDs and item IDs
    combinations = list(product(selected_user_ids, selected_item_ids))
    
    # Create a new DataFrame with all combinations
    data_entropy0 = pd.DataFrame(combinations, columns=['user_id', 'item_id'])
    
    # Merge with the original DataFrame to retain existing raw_ratings values
    data_entropy0 = data_entropy0.merge(data_pd[data_pd['item_id'].isin(threshold_item_df)], on=['user_id', 'item_id'], how='left')
    
    # Fill missing raw_ratings with 0
    data_entropy0['raw_ratings'].fillna(0, inplace=True)
    
    # drop index
    data_entropy0.reset_index(drop=True, inplace=True)
    
    unique_item_id = selected_item_ids
    entropy_list = np.empty(shape=(len(unique_item_id), 2), dtype=object)
    j = 0
    
    # loop over all itemId's and compute the entropy for each item
    for i in unique_item_id:
        item_i_df = data_entropy0[data_entropy0['item_id'] == i]
        entropy_list[j] = [i, entropy(item_i_df['raw_ratings'])]
        j += 1
    
    # transform to pandas DataFrame
    to_df = {'item_id' : entropy_list[:,0],'entropy0' : entropy_list[:,1]}
    ent_items_df_k = pd.DataFrame(to_df)
    ent_items_df_k.sort_values(by='entropy0',inplace=True,ascending=False)
    
    print('Computed entropy0 scores for all items')
    
    ### ENTROPY0 POPULARITY STRATEGY ###
    # prepare item entropies for merging
    item_freq_entropy0 = item_freq_df[item_freq_df['item_id'].isin(ent_items_df_k['item_id'])]
    # merge frequencies and entropies
    entpop_items_df_k = pd.merge(item_freq_entropy0, ent_items_df_k, on='item_id')
    
    # compute popent score
    entpop_items_df_k['entpop'] = 0.6*np.log10(entpop_items_df_k['freq'])+0.2*entpop_items_df_k['entropy0']
    entpop_items_df_k.sort_values(by='entpop',inplace=True,ascending=False)

    print('Computed entpop scores for all items')
    
    # Testing
    item_to_be_shown = [10, 25, 50, 100]
    
    for nr_of_shown_items in item_to_be_shown:
        print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))
        number_of_shown_items = str(nr_of_shown_items)
        
        ### ENTROPY0 STRATEGY ###
        # select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest entropy
        ent_items = ent_items_df_k.head(nr_of_shown_items)['item_id']
        ent_items = np.array(ent_items)
        print('Computed ranking using entropy0 strategy')
        
        # ENTPOP STRATEGY
        # select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
        entpop_items = entpop_items_df_k.head(nr_of_shown_items)
        entpop_items = np.array(entpop_items['item_id'])
        print('Computed ranking using entropy0 popularity strategy')
        
        print('Computing results')
        
        ### ENTROPY0 STRATEGY ###
        ranking_strategy = 'Entropy0 strategy'
        # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
        train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(ent_items))]
        test_pd = data_pd[(data_pd.user_id.isin(cold_users)) & (~data_pd.item_id.isin(ent_items))]
        
        train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                    reader = Reader(rating_scale=(0, 1)))
        
        test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
        
        # Retrieve the trainset.
        train = train.build_full_trainset()
        
        # fit model
        model.fit(train)
        
        # predict
        pred = model.test(test)
        
        # compute RMSE
        rmse = accuracy.rmse(pred)
        
        # store result in csv file
        writer.writerow([ranking_strategy, number_of_shown_items, 
                            nr_of_cold_users, rmse, k])
        
        print('RMSE computed for entropy0 strategy')
        
        # store result in csv file
        writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse, k])
        
        # ENTPOP STRATEGY
        ranking_strategy = 'Entropy0 popularity strategy'
        # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
        train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(entpop_items))]
        test_pd = data_pd[(data_pd.user_id.isin(cold_users)) & (~data_pd.item_id.isin(entpop_items))]
        train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                           reader = Reader(rating_scale=(0, 1)))
        test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
        # Retrieve the trainset.
        train = train.build_full_trainset()
        # fit model
        model.fit(train)
        # predict
        pred = model.test(test)
        # compute RMSE
        rmse = accuracy.rmse(pred)
        print('RMSE computed for entropy0 popularity strategy')
        # store result in csv file
        writer.writerow([ranking_strategy, number_of_shown_items, 
                            nr_of_cold_users, rmse, k])

csvfile.close()   
