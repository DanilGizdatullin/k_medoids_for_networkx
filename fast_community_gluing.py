import networkx as nx
import pickle
import random
import numpy as np
import sys
import time

from collections import OrderedDict


class FastCommunityGluing:
    def __init__(self, graph=nx.Graph(), max_parameter=1):
        self.graph = graph
        self.max_parameter = float(max_parameter)
        self.node_clusters = {}
        self.cluster_nodes = {}
        self.clusters_by_level = []
        self.edges_sort_by_affinity = []
        free_cluster = 0
        for i in self.graph.nodes():
            self.node_clusters[i] = free_cluster
            self.cluster_nodes[free_cluster] = [i]
            free_cluster += 1
        self.clusters_by_level.append(self.node_clusters)

        edges = self.graph.edges()
        edges_with_affinity = {}
        for i in edges:
            # edges_with_affinity[(i[0], i[1])] = self.graph[i[0]][i[1]]['a'] / self.max_parameter
            edges_with_affinity[(i[0], i[1])] = self.graph[i[0]][i[1]]['a']
        edges_with_affinity = OrderedDict(sorted(edges_with_affinity.items(), key=lambda t: -t[1]))

        self.edges_sort_by_affinity = edges_with_affinity.keys()

        # print(self.cluster_nodes)
        # print(self.node_clusters)

    def dendrogram(self, initial_state=None):
        print(len(set(self.clusters_by_level[0].values())))
        if initial_state:
            print('Initial State')
            self.node_clusters = initial_state
            self.cluster_nodes = {}
            set_of_clusters = set(self.node_clusters.values())
            for i in set_of_clusters:
                self.cluster_nodes[i] = []
            for key, value in self.node_clusters.iteritems():
                self.cluster_nodes[value].append(key)
        for i in self.edges_sort_by_affinity:
            node1 = i[0]
            node2 = i[1]
            # print(self.graph[node1][node2]['a'])
            if node1 != node2:
                cluster_node1 = self.node_clusters[node1]
                cluster_node2 = self.node_clusters[node2]
                # print('Cluster 1 = {}, Cluster 2 = {}'.format(cluster_node1, cluster_node2))
                if cluster_node1 != cluster_node2:
                    nodes_from_old_cluster = self.cluster_nodes[cluster_node2]
                    for node in nodes_from_old_cluster:
                        self.cluster_nodes[cluster_node1].append(node)
                        self.node_clusters[node] = cluster_node1
                    del self.cluster_nodes[cluster_node2]
                # cluster_node1 = self.node_clusters[node1]
                # cluster_node2 = self.node_clusters[node2]
                # print('Cluster 1 = {}, Cluster 2 = {}'.format(cluster_node1, cluster_node2))
                # print(len(set(self.node_clusters.values())))
            self.clusters_by_level.append(self.node_clusters.copy())
        pickle.dump(self.clusters_by_level, open('list_of_clusters_by_level.p', 'wb'))

if __name__ == '__main__':
    books_graph = nx.read_gpickle('example_data/books_graph_main_connected_part.p')
    print('Done!')
    step1 = time.time()
    # clustering = FastCommunityGluing(graph=books_graph, max_parameter=2921242)
    # clustering.dendrogram()
    print(time.time() - step1)

    # time_start = time.time()
    # clusters_by_level = pickle.load(open('list_of_clusters_by_level.p', 'rb'))
    # print('Done!')
    # print(time.time() - time_start)
    # number_of_levels = len(clusters_by_level)
    # for i in xrange(number_of_levels):
    #     print('Level {}, Number_of_clusters = {}'.format(i, len(set(clusters_by_level[i].values()))))
    # print(clusters_by_level[8269805])
