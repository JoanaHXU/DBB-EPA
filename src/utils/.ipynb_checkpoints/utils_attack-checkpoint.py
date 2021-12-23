import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

''' Cosine Function '''
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]
        

""" To measure attack_cost from < EMBEDDINGs > """
def Attack_Cost_Done(env, zA, zB, E, E_init, done_threshold=0.01, weighting_para = 0.1):

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
    
    # done or not?
    if Z_deviation < done_threshold:
        done = 1
    else:
        done = 0
        
    success_rate = Z_cosine

    return cost, done, success_rate


""" To measure attack_cost from < A BATCH OF EMBEDDINGs > """
def Attack_Cost_Done_batch(env, zA, zB, E, E_init, done_threshold=0.01, weighting_para = 0.1):
    n_embedding = len(zA)
#     print(f"N_embedding = {n_embedding}")

    # cosine similairy of Z
    Z_cosine = 0
    for i in range(n_embedding):
        similarity =  get_cosine_similarity(zA[i],zB[i])
#         print(f"similarity = {similarity}")
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
    
    # done or not?
    if Z_deviation <= done_threshold:
        done = 1
    else:
        done = 0
        
    success_rate = Z_avg_cosine

    return cost, done, success_rate
    
    
def Deviation_of_Z(zA, zB):
    # deviation between embeddings, via cosine_similarity
    Z_cosine =  get_cosine_similarity(zA,zB)
    Z_deviation = 1 - Z_cosine
    
    return Z_deviation
    
    