import numpy as np
import csv

# path for buzzfeed user/user info
path1 = 

# path for politifact user/user info
path2 = 

# path for buzzfeed user ids
path3 = 

# path for politifact user ids
path4 = 

# path for buzzfeed article/user info
path5 = 

# path for politifact article/user info
path6 = 

# output matrix path
path_output = 

u_u_1 = np.loadtxt(path1, dtype=int)
u_u_2 = np.loadtxt(path2, dtype=int)

user_ids = list(np.loadtxt(path3, dtype=str))
user_ids_pol = list(np.loadtxt(path4, dtype=str))

for line in user_ids_pol:
    if line not in user_ids:
        user_ids.append(line)

user_ids_buz = list(np.loadtxt(path3, dtype=str))

user_dict = dict()
for idx, obj in enumerate(user_ids):
    user_dict[obj] = idx

user_dict_buzz = dict()
for idx, obj in enumerate(user_ids_buz):
    user_dict_buzz[idx] = obj
    
user_dict_pol = dict()
for idx, obj in enumerate(user_ids_pol):
    user_dict_pol[idx] = obj
    
uu1 = np.zeros((len(u_u_1), 2))
uu2 = np.zeros((len(u_u_2), 2))

u_u_1 = u_u_1 - 1
u_u_2 = u_u_2 - 1

for i in range(len(u_u_1)):
    for j in range(2):
        uu1[i,j] = user_dict[user_dict_buzz[u_u_1[i,j]]]
        
for i in range(len(u_u_2)):
    for j in range(2):
        uu2[i,j] = user_dict[user_dict_pol[u_u_2[i,j]]]

adj_user = np.zeros((len(user_dict), len(user_dict)))

for idx, obj in enumerate(uu1):
    adj_user[int(obj[0]), int(obj[1])] = 1
    
for idx, obj in enumerate(uu2):
    adj_user[int(obj[0]), int(obj[1])] = 1
    
###############################################################################
    
user_art_buz = (np.loadtxt(path5, dtype=int))
user_art_pol = (np.loadtxt(path6, dtype=int))

user_art_buz[:,1] = user_art_buz[:,1] - 1
user_art_pol[:,1] = user_art_pol[:,1] - 1

for i in range(len(user_art_buz)):
    user_art_buz[i,1] = user_dict[user_dict_buzz[user_art_buz[i,1]]]
    
for i in range(len(user_art_pol)):
    user_art_pol[i,1] = user_dict[user_dict_pol[user_art_pol[i,1]]]
    
unique_art_buz = []
unique_art_pol = []

for i in range(len(user_art_buz)):
    if user_art_buz[i,0] not in unique_art_buz:
        unique_art_buz.append(user_art_buz[i,0])

for i in range(len(user_art_pol)):
    if user_art_pol[i,0] not in unique_art_pol:
        unique_art_pol.append(user_art_pol[i,0])
        
art_dict_buz = dict()
for idx, obj in enumerate(unique_art_buz):
    art_dict_buz[obj] = idx

art_dict_pol = dict()
for idx, obj in enumerate(unique_art_pol):
    art_dict_pol[obj] = idx + 182

adj_art_user = np.zeros((240+182, 35321))
for i in range(len(user_art_buz)):
    adj_art_user[art_dict_buz[user_art_buz[i,0]], user_art_buz[i,1]] = user_art_buz[i,2]
for i in range(len(user_art_pol)):
    adj_art_user[art_dict_pol[user_art_pol[i,0]], user_art_pol[i,1]] = user_art_pol[i,2]
    
adj_art_user2 = np.zeros((35321, 422))
for i in range(len(user_art_buz)):
    adj_art_user2[user_art_buz[i,1], art_dict_buz[user_art_buz[i,0]]] = user_art_buz[i,2] 
for i in range(len(user_art_pol)):
    adj_art_user2[user_art_pol[i,1], art_dict_pol[user_art_pol[i,0]]] = user_art_pol[i,2]

adj = np.concatenate((adj_user, adj_art_user), axis = 0)

adj_fill = np.zeros((422,422))

adj2 = np.concatenate((adj_art_user2, adj_fill), axis = 0)

adj_matrix = np.concatenate((adj, adj2), axis = 1)

adj_matrix_dict = dict()
for key, val in user_dict.items():
    adj_matrix_dict[val] = key
    
for key, val in art_dict_buz.items():
    adj_matrix_dict[val+35322] = 'buzz'+str(key)
    
for key, val in art_dict_pol.items():
    adj_matrix_dict[val+35322] = 'pol'+str(key)

for i in range(len(adj)):
    adj[i,i] = 1
    
from scipy import sparse 
sadj = sparse.csr_matrix(adj)

sparse.save_npz(path_output,sadj)

sparse_matrix = sparse.load_npz(path_output)
        
        
