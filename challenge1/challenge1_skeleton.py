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

def list_of_different_attribute_values(d):
    return set([v for values in d.values() for v in values])
    
def draw_graph(g, node_attribute=None, list_of_values_of_attributes=None):
    """
    Draw the graph g.

    Parameters
    ----------
    g : graph
       A networkx graph
    node_attribute : string 
       The name of the node attribute used to assign colors to the drawing
    list_of_values_of_attributes : list
        A list of all the potential values of node_attribute to assign one color
        per value.
    """
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    
    pos = nx.spring_layout(g, iterations=100)

    
    if node_attribute and list_of_values_of_attributes: 
        # To associate colors to nodes according to an attribute, here college
        # build a color_map, one for each college
        color_map={}
        i=0.0
        for s in list_of_values_of_attributes:
            color_map[s]=i
            i+=1/len(list_of_values_of_attributes)
        color_map[None]=1 # for nodes without values for the attribute node_attribute
        
        # The values supplied to node_color should be in the same order as the nodes 
        # listed in G.nodes(). We take an arbitrary mapping of values color_map and 
        # generate the values list in the correct order
        #values = [color_map[G.node[node].get(node_attribute)] for node in G.nodes()] # for attributes encoded in the graph
        values=[]        
        for node in G.nodes():
            if node in node_attribute:
                if node_attribute[node]:
                    # we arbitrarily take the first value 
                    values.append(color_map[node_attribute[node][0]])
            else:
                values.append(1)
               
        nx.draw_networkx_nodes(g,pos, cmap=plt.get_cmap('jet'), node_color=values)
    else:
        nx.draw_networkx_nodes(g,pos)
       
    nx.draw_networkx_edges(g,pos)
    nx.draw_networkx_labels(g,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.show()
    pylab.close()
    del fig

def properties(g):
    """
    Computes simple and classic graph metrics.

    Parameters
    ----------
    g : graph
       A networkx graph
    """
    # networkx short summary of information for the graph g
    print(nx.info(g))
    
    # Draw the degree distribution. Powerlow distribution for a real (complex) network
    plt.figure(num=None)
    fig = plt.figure(1)
    degree_sequence=list(dict(nx.degree(g)).values()) # degree sequence
    print("Degree sequence %s" % degree_sequence)
    plt.hist(degree_sequence, bins='auto')  
    plt.title("powerlaw degree distribution")
    plt.ylabel("# nodes")
    plt.xlabel("degree")
    plt.show()
    pylab.close()
    del fig
    
    precomputed_eccentricity = nx.eccentricity(g) # costly step, we save time here!
    print("Graph density %f" % nx.density(g))
    print("Diameter (maximum eccentricity): %d" % nx.diameter(g,precomputed_eccentricity))
    print("Radius (minimum eccentricity): %d" % nx.radius(g,precomputed_eccentricity)) #The radius is the minimum eccentricity.
    print("Mean eccentricity (eccentricity(v) = the maximum distance from v to all other nodes): %s" % np.mean(list(precomputed_eccentricity.values())))
    print("Center is composed of %d nodes (nodes with eccentricity equal to radius)" % len(nx.center(g, precomputed_eccentricity)))
    print("Periphery is composed of %d nodes (nodes with eccentricity equal to the diameter)" % len(nx.periphery(g,precomputed_eccentricity)))
    print("Mean clustering coefficient %f" % np.mean(list(nx.clustering(g).values())))
    total_triangles=sum(nx.triangles(g).values())/3    
    print("Total number of triangles in graph: %d" % total_triangles)



# --------------------- Let's take a quick NetworkX tour ------------------------------#
# load the graph
G = nx.read_gexf("mediumLinkedin.gexf")

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes, or just location, or all of same
# Some nodes may have 2 or more colleges or employers, so we
# use dictionaries to store the attributes
college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer.pickle', 'rb') as handle:
    employer = pickle.load(handle)

print("Nb of users with one or more attribute college: %d" % len(college))
print("Nb of users with one or more attribute location: %d" % len(location))
print("Nb of users with one or more attribute employer: %d" % len(employer))

#print(list_of_different_attribute_values(employer))

# Let's have a look on the graph
#draw_graph(G, node_attribute=employer, list_of_values_of_attributes=list_of_different_attribute_values(employer))

# print some properties to understand the type of graph
#properties(G)

print("\n\n")
# to see the nodes with the highest degree
# We use sorted with as parameter the list of items (key + value) from our dict 
# The sort key is a lambda function which gives the indice of the data to sort.
# t is a tuple (key, value) from the dict
# Lambda says to sort with the value of the tuple: t[1] 
# to sort the tuple on the key, choose t[0]
sorted_nodes_by_degree=sorted(dict(nx.degree(G)).items(), key=lambda t: t[1])
print("The 5 nodes with the highest degree are: %s" % str(sorted_nodes_by_degree[-5:]))
print("5 of the nodes with the smallest degree are: %s" % str(sorted_nodes_by_degree[:5]))
print("\n\n")

# To see the neighbors of a node
print("U7912 has %d neighbors" % len(list(G.neighbors('U7912'))))
print("U7912 has %d neighbors" % G.degree('U7912'))

if 'U7912' in college:
    print("U7912's college(s): %s" % str(college['U7912']))
print("Her/his neighbors'colleges are:")
for n in G.neighbors('U7912'):
    if n in college:
        print("--- U7912's neighbor %s went in college(s): %s" % (n,str(college[n])))
    else:
        print("--- %s has not filled his/her college profil" % str(n))
print("\n\n")

# Fast examination of all edges is achieved using adjacency iterators.
# Useful to explore the neighbors of each node
# Here, we compute the homophily according to the attribute college
similar_neighbors=0
total_number_neighbors=0 # to verify the number of edges ;-)!!!
for n,nbrs in G.adjacency():
    for nbr in nbrs:
        total_number_neighbors+=1
        if n in college and nbr in college:
            if len([val for val in college[n] if val in college[nbr]]) > 0:
                similar_neighbors+=1
homophily=similar_neighbors/total_number_neighbors
print("Homophily ('college' attribute), i.e. total proportion of neighbors sharing attributes: %f" % homophily)
print('Is our number of edges (=%d) similar to networkX property (=%d)? No? Normal! why?' % (total_number_neighbors, len(G.edges())))

print("\n\nNow your turn!!! ")

# --------------------- Now your turn -------------------------------------#
# Lest's us say you are U19886 and you want to work at Google.
# Explore, implement your strategy to be helped in getting this job at Google.
sommet='U19886' #sommet is in location, not in location neither in employer
Neighbors=list(G.neighbors('U19886'))


def similaire(n1,n2):
	toReturn=100
	if(n1 in college and n2 in college):
		toReturn-=60*(len([val for val in college[n1] if val in college[n2]]) > 0)
	if (n1 in location and n2 in location):
		toReturn-=20*(len([val for val in location[n1] if val in location[n2]]) > 0)
	if (n1 in employer and n2 in employer):
		toReturn-=10*(len([val for val in employer[n1] if val in employer[n2]]) > 0)
	return toReturn

#print(location[sommet], len(googleWorkers))
#for n in googleWorkers:
	#print("%s et %s ont %d similarités \n" %(sommet,n,similaire(sommet,n)))
#for  nbr in Neighbors:
#	print("%s et %s ont %d similarités \n" %(sommet,nbr,similaire(sommet,nbr)))
for (u,v) in list(G.edges()):
	G.add_edge(u,v,weight=similaire(u,v))
#print(G.edges(data= True))




#################################################### LES EMPLOYERS DE GOOGLE ###########################################
googleWorkers=[]
for n,nbrs in G.adjacency():
    if (n in employer) and ('google' in employer[n]):
        googleWorkers.append(n)
"""
        dij=list(nx.dijkstra_path(G,sommet,n))
        G.add_path(dij)
        print(nx.shortest_path(G,sommet,n))
        print(nx.dijkstra_path(G,sommet,n))

"""

#################################################### METTRE DES POIDS SUR LES LIENS CONFORMÉMENT AUX LIENS QUE PARTAGENT LES EXTREMITES ################################# 
for (u,v) in G.edges():
    G.add_edge(u, v, weight=similaire(u,v) )





list_path_length=[]
#################################################### RECUPERER POUR CHAQUE EMPLOYE GOOGLE LE PLUS COURT CHEMIN ET SA LONGUEUR PARTANT  DE LA PERSONNE U19886  ######################################
for employe in googleWorkers :
    list_path_length.append(nx.dijkstra_path_length(G, sommet, employe,weight='weight'))

#print(list_path_length)

#################################################### Les cinq chemins les plus courts menant vers un employé de google avec des relations privilégiées #############################################
def les_minima(L,n):
    M=[]
    if len(L)>=n:
    	i=0
    	while i<n:
    		m=min(L)
    		if not m in M:
    			M.append(m)
    			i+=1
    		L.remove(m)
    print(M)
    return M

M= les_minima(list_path_length,5)
seuil = min(M)

for employe in googleWorkers :
    if nx.dijkstra_path_length(G, sommet, employe,weight='weight') <= seuil :
        print(nx.dijkstra_path(G, sommet, employe,weight='weight'))
        for n in nx.dijkstra_path(G, sommet, employe,weight='weight'):
        	if n in college:
        		print(n,' est allé à ', college[n])
        	if n in location:
        		print(n,' a habité à ', location[n])
        	if n in employer:
        		print(n,' a travaillé chez ', employer[n])
        	print('\n\n')
    #if nx.shortest_path_length(G, sommet, employe,weight='weight') <= seuil :
        #print(nx.shortest_path(G, sommet, employe,weight='weight'))