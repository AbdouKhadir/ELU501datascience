# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:09:11 2017

@author: cbothore
"""


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter



def naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node (from empty), value is a list of attribute values. Here 
       only 1 value in the list.
     """
     
    predicted_values={}
    for n in empty:
        nbrs_attr_values=[]
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values

def max_clique(C): #C is a list of list of nodes. It represents the list of cliques in the neighborhood of an empty_node and will be used the clique_method
    toReturn=[]
    if C:
        for L in C:
            if len(L)>len(toReturn):
                toReturn=L
    return toReturn

def coeff_method(graph, empty, attr):
    predicted_values={}
    for n in empty:
        nbrs_attr_values=[]
        SG=nx.subgraph(graph, list(graph.neighbors(n))+[n])
        liste_cliques=list(nx.find_cliques(SG))
        iter=[]
        clique=max_clique(liste_cliques)
        if len(clique)>4:
            iterListe=clique
        else:
            iterListe=graph.neighbors(n)
        for nbr in iterListe:
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)

            #define share coefficient for each attribute value
            
            coeff_dict={key: value for key,value in cpt.items()}
            for val_attr in coeff_dict:
                number_of_share=1
                for nbr1 in SG.nodes():
                    for nbr2 in SG.nodes():
                        if (nbr2!=nbr1) and (nbr1 in attr ) and (nbr2 in attr):
                            number_of_share+= (val_attr in attr[nbr1]) and (val_attr in attr[nbr2])
                coeff_dict[val_attr]=cpt[val_attr]*number_of_share

            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(coeff_dict.items(), key=lambda t: t[1])
            predicted_values[n].append([a,nb_occurrence])
    return predicted_values

def naive_method2(graph, empty, attr):
    
     
    predicted_values={}
    for n in empty:
        nbrs_attr_values=[]
        #SG=nx.subgraph(graph, list(graph.neighbors(n))+[n])
        #clique=max_clique(list(SG.nodes()).remove(n))
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values


def attribut_exhaustive_method(graph, empty, all_attr):
    all_attr_predicted_values={}
    for attr in all_attr:
        all_attr_predicted_values[attr]=coeff_method(graph,empty,all_attr[attr])

    for n in empty:
        maxi=0
        for attr in all_attr_predicted_values:
            #print(all_attr_predicted_values[attr][n])
            if all_attr_predicted_values[attr][n] and maxi<all_attr_predicted_values[attr][n][0][1]:
                maxi=all_attr_predicted_values[attr][n][0][1]
        for attr in all_attr_predicted_values:
                    if all_attr_predicted_values[attr][n] and maxi>all_attr_predicted_values[attr][n][0][1]:
                        all_attr_predicted_values[attr][n]=[]
                    elif all_attr_predicted_values[attr][n]:
                        all_attr_predicted_values[attr][n]=[all_attr_predicted_values[attr][n][0][0]]

    return all_attr_predicted_values #dico de dico




    
 
def evaluation_accuracy(groundtruth, pred):
    """    Compute the accuracy of your model.

     The accuracy is the proportion of true results.

    Parameters
    ----------
    groundtruth :  : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.
    pred : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values. 

    Returns
    -------
    out : float
       Accuracy.
    """
    true_positive_prediction=0   
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction+=1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            true_positive_prediction += len([c for c in p_value if c in groundtruth[p_key]])/len(groundtruth[p_key])        
        # no else, should not happen: train and test datasets are consistent
    return 100*true_positive_prediction/len(pred)
   

# load the graph
G = nx.read_gexf("mediumLinkedin.gexf")
print("Nb of users in our graph: %d" % len(G))

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes
# Some nodes may have 1 attribute 'location'
# Some nodes may have 1 or more 'colleges' or 'employers', so we
# use dictionaries to store the attributes
college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)

print("Nb of users with one or more attribute college: %d" % len(college))
print("Nb of users with one or more attribute location: %d" % len(location))
print("Nb of users with one or more attribute employer: %d" % len(employer))

# here are the empty nodes for whom your challenge is to find the profiles
empty_nodes=[]
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
print("Your mission, find attributes to %d users with empty profile" % len(empty_nodes))


# --------------------- Baseline method -------------------------------------#
# Try a naive method to predict attribute
# This will be a baseline method for you, i.e. you will compare your performance
# with this method
# Let's try with the attribute 'employer'
attribut_group={'college':college, 'location':location, 'employer':employer}

print('FOR LOCATION')

location_predictions=attribut_exhaustive_method(G, empty_nodes, attribut_group)['location']
groundtruth_location={}
with open('mediumLocation.pickle', 'rb') as handle:
    groundtruth_location = pickle.load(handle)
result=evaluation_accuracy(groundtruth_location,location_predictions)
print("%f%% of the predictions are true" % result)

print('FOR EMPLOYER')
employer_predictions=attribut_exhaustive_method(G, empty_nodes, attribut_group)['employer']
groundtruth_employer={}
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
result=evaluation_accuracy(groundtruth_employer,employer_predictions)
print("%f%% of the predictions are true" % result)

print('FOR COLLEGE')
college_predictions=attribut_exhaustive_method(G, empty_nodes, attribut_group)['college']
groundtruth_college={}
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_college = pickle.load(handle)

result=evaluation_accuracy(groundtruth_college,college_predictions)
print("%f%% of the predictions are true" % result)



# def compteur(dico):
#     somme=0
#     for key in dico:
#         if dico[key]:
#             somme+=1
#     return somme
# print({key:compteur(deeper_naive_method(G, empty_nodes, attribut_group)[key]) for key in attribut_group})
# print({key:len(deeper_naive_method(G, empty_nodes, attribut_group)[key]) for key in attribut_group})
#print(deeper_naive_method(G,empty_nodes,attribut_group)['location'])
# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes

# and compare with the ground truth (what you should have predicted)
# user precision and recall measures
def vect_model(graph,Vect,attr_group):
    #Vect est une liste vide
    #attr_group est un dictionaire {node:[attr1,attr2,...]} comme employer, location ou college7
    #cette fonction recense dans une liste toutes les valeurs d'attributs possibles  contenues dans attr_group et insere cette liste à la fin de Vect
    Vect.append([])
    for node in graph:
        if node in attr_group:
            for attr_value in attr_group[node]:
                if not attr_value in Vect[-1]:
                    Vect[-1].append(attr_value)


Model=[]
vect_model(G,Model,employer)
vect_model(G,Model,location)
vect_model(G,Model,college) # Model=[[employer1,employer2,...],   [location1,location2,...],    [college1, college2,.....]]
#print([len(Model[i]) for i in range(3)])

#def attr_vect(graph, attr_group):

# print("STATICS")
# stats=naive_method(G,groundtruth_location,groundtruth_location)
# result=evaluation_accuracy(groundtruth_location,stats)
# print("%f%% of the predictions are true" % result)
