import numpy as np


genre_list = ['Animation',
              'Children\'s',
              'Comedy',
              'Adventure',
              'Action',
              'Crime',
              'Thriller',
              'Romance',
              'Drama',
              'Horror',
              'Fantasy',
              'Sci-Fi',
              'Documentary',
              'War',
              'Musical',
              'Mystery',
              'Western',
              'Film-Noir'
              ]

with open('data_sets/users.dat', 'r', encoding='latin-1') as f:
    d = f.readlines()
    user_data_set = [i.rstrip().split("::") for i in d]

with open('data_sets/ratings.dat', 'r', encoding='latin-1') as f:
    d = f.readlines()
    rating_data_set = [i.rstrip().split("::") for i in d]

with open('data_sets/movies.dat', 'r', encoding='latin-1') as f:
    d = f.readlines()
    movie_data_set = [i.rstrip().split("::") for i in d]


def context_vectors(movie_data):
    '''Construct context vectors from movie data.'''
    genre_string = np.asarray(movie_data)[:, -1]
    context_data = np.zeros((len(genre_string),
                             len(genre_list)))
    for i in range(0, len(genre_string)):
        genres = genre_string[i].split('|')
        for j in range(0, len(genre_list)):
            if genre_list[j] in genres:
                context_data[i][j] += 1
        context_data[i] /= np.sum(context_data[i])

    return context_data

def bandit_rewards(user_data, rating_data, movie_data):
    '''construct reward matrix, based on user ratings.'''
    reward_data = np.zeros((len(user_data), len(movie_data)))
    for k in range(0, len(rating_data)):
        if int(rating_data[k][1]) > len(movie_data):
            continue

        reward_data[int(rating_data[k][0])-1][int(rating_data[k][1])-1] = int(rating_data[k][2])
    return reward_data


def extract_context_for_users(user_index, context_data, rating_data):
    '''Extract available context vectors for specific users.'''
    user_indices = np.where(rating_data[user_index] != 0)
    return context_data[user_indices], rating_data[user_index][user_indices]


def filter_users(user_data, gender=None, age=None, prof=None):
    '''Filter users by specific traits.'''
    usr_dta = user_data.copy()
    if gender is not None:
        gender_ind = np.where(usr_dta[:, 1]==gender)
        usr_dta = usr_dta[gender_ind]

    if age is not None:
        if age < 18:
            age_string = '1'

        elif 18 <= age <= 24:
            age_string = '18'

        elif 25 <= age <= 34:
            age_string = '25'

        elif 35 <= age <=44:
            age_string = '35'

        elif 45 <= age <= 49:
            age_string = '45'

        elif 50 <= age <= 55:
            age_string = '50'

        else:
            age_string = '56'

        age_ind = np.where(usr_dta[:, 2]==age_string)
        usr_dta = usr_dta[age_ind]

    if prof is not None:
        prof_ind = np.where(usr_dta[:, 3]==prof)
        usr_dta = usr_dta[prof_ind]

    return usr_dta


context_data_set = context_vectors(movie_data_set)
reward_data_set = bandit_rewards(user_data_set, rating_data_set, movie_data_set)
