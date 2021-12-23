import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


""" To measure attack_cost from < EMBEDDINGs > """
def Attack_Cost_SINGLE(env, zA, zB, E, E_init, weighting_para = 0.1):

    # deviation between embeddings, via cosine_similarity
    Z_cosine =  get_cosine_similarity(zA,zB)
    Z_deviation = 1 - Z_cosine
    
    # deviation between envs, via euclidean distance
    E_l2 = np.linalg.norm(E - E_init)
    E_max = np.ones(len(E))*env.hyper_max
    E_l2_max = np.linalg.norm(E_max - E_init)
    E_deviation = E_l2/E_l2_max
    
    # weighted cost
    cost = (1-weighting_para)*Z_deviation + weighting_para*E_deviation

    return cost


""" To measure attack_cost from < A BATCH OF EMBEDDINGs > """
def Attack_Cost_BATCH(env, zA, zB, E, E_init, weighting_para = 0.1):
    
    n_embedding = len(zA)

    # cosine similairy of Z
    Z_cosine = 0
    for i in range(n_embedding):
        similarity =  get_cosine_similarity(zA[i],zB[i])
        Z_cosine += similarity
    Z_avg_cosine = Z_cosine/n_embedding
    Z_deviation = 1 - Z_avg_cosine

    # deviation between envs, via euclidean distance
    E_l2 = np.linalg.norm(E - E_init)
    E_max = np.ones(len(E))*env.hyper_max
    E_l2_max = np.linalg.norm(E_max - E_init)
    E_deviation = E_l2/E_l2_max

    # cost： weighted cost
    cost = (1-weighting_para)*Z_deviation + weighting_para*E_deviation

    return cost



"""
Function to measure whether attack done
"""
def Attack_Done(env, target, Q):
    error = 0
    amount = len(target)

    for s in range(env.nS):
        if s in target:
            target_a_index = np.argmax(target[s])
            learner_a_index = np.argmax(Q[s])

            if target_a_index != learner_a_index:
                error += 1

    if error == 0:
        done = 1
    else:
        done = 0

    accuracy_rate = (amount-error)/amount

    return done, accuracy_rate