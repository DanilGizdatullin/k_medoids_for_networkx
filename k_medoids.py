import networkx as nx
import pickle
import random
import numpy as np
import sys
import time


class KMedoids:
    def __init__(self, graph=nx.Graph(), k=2, max_parameter=1):
        self.graph = graphgit
        self.number_of_classes = k
        self.max_parameter = float(max_parameter)
        self.distance_between_nodes = {}
        self.node_cluster = {}
        self.cluster_nodes = {}
        self.nodes = list(self.graph.nodes())
        self.medoids = np.ndarray(k)
        self.nearest_centroids = {}
        for i in xrange(k):
            self.nearest_centroids[i] = []

    # def find_nearest_clusters(self):
    #     for i in xrange(len(self.medoids)):
    #         for j in

    def find_distance(self, source=0, target=1):
        """
        Find distance as a production of scaling affinity of edges in shortest path.
        The bigger value for more similarity.

        :param source: int: node1
        :param target: int: node2
        :return:  float: distance between 0 and 1
        """

        if source == target:
            return 1.0
        elif self.graph.has_edge(source, target):
            return self.graph[source][target]['a'] / self.max_parameter
        else:
            min_node = min(source, target)
            max_node = max(source, target)

            if (min_node, max_node) not in self.distance_between_nodes:
                path = nx.shortest_path(self.graph, source=source, target=target)
                mult = 1
                for i in xrange(1, len(path)):
                    mult *= (self.graph[path[i-1]][path[i]]['a'] / self.max_parameter)
                self.distance_between_nodes[(min_node, max_node)] = mult
                return mult
            else:
                return self.distance_between_nodes[(min_node, max_node)]

    def recompute_clusters(self, medoids=np.ndarray(2)):
        """
        Find clusters based on medoids and special metric.

        :param medoids: numpy array of medoids
        :return: two dictionaries. First - {node_id: cluster_id}, Second - {cluster_id: [list of nodes]}
        """
        dict_of_clusters = {}
        reversed_dict_of_clusters = {}

        for i in xrange(self.number_of_classes):
            reversed_dict_of_clusters[i] = []
        for n_i in self.nodes:
            path_length = np.ndarray(self.number_of_classes)
            for j in xrange(self.number_of_classes):
                length = self.find_distance(n_i, medoids[j])
                path_length[j] = length
            # print('========')
            # print(n_i)
            # print(path_length)
            idx = np.argmax(path_length)
            dict_of_clusters[n_i] = idx
            reversed_dict_of_clusters[idx].append(n_i)

        return dict_of_clusters, reversed_dict_of_clusters

    def quality_function(self, dict_of_clusters={}, medoids=np.ndarray(2)):
        """
        Compute the quality of clusterization. The sum of closeness between medoids and their nodes.

        :param dict_of_clusters: {node_id: cluster_id}
        :param medoids: numpy array of cluster medoids
        :return: float of clusters quality
        """
        quality = 0
        for node_i, cluster_i in dict_of_clusters.iteritems():
            quality += self.find_distance(source=node_i, target=medoids[cluster_i])

        return quality

    def k_medoids(self, border_of_stable=2, s=2):
        """
        Compute clusters.

        :param border_of_stable: how many steps in the same state to stop algorithm
        :param s: how many new candidates from the same class
        :return: dictionary of nodes and classes, medoids
        """

        # nodes = np.fromiter(np.array(self.nodes), np.int)
        nodes = np.array(self.nodes)
        # num = 1
        self.medoids = np.random.choice(nodes, size=self.number_of_classes, replace=False)
        self.node_cluster, self.cluster_nodes = self.recompute_clusters(self.medoids)
        main_quality = self.quality_function(self.node_cluster, self.medoids)
        print('First Approximation')
        print('Quality = {}'.format(main_quality))
        print('Medoids: {}'.format(self.medoids))
        print('Dict of clusters: {}'.format(self.node_cluster))
        print('='*35)
        # # the main iterative procedure
        stable_sequence = 0
        while True:
            for j in range(self.number_of_classes):
                print('Iteration')
                max_quality_for_current_cluster = main_quality
                medoids_with_max_quality_for_current_cluster = []
                dict_of_clusters_with_max_quality_for_current_cluster = []
                dict_of_reversed_clusters_with_max_quality_for_current_cluster = []

                if len(self.cluster_nodes[j]) > s:
                    s_nodes = np.random.choice(self.cluster_nodes[j], size=s, replace=False)
                    for v in s_nodes:
                        temp_medoids = self.medoids.copy()
                        temp_medoids[j] = v
                        temp_clusters, temp_reverse_clusters = self.recompute_clusters(temp_medoids)
                        temp_quality = self.quality_function(temp_clusters, temp_medoids)
                        if temp_quality > max_quality_for_current_cluster:
                            max_quality_for_current_cluster = temp_quality
                            medoids_with_max_quality_for_current_cluster = temp_medoids
                            dict_of_clusters_with_max_quality_for_current_cluster = temp_clusters
                            dict_of_reversed_clusters_with_max_quality_for_current_cluster = temp_reverse_clusters

                    if max_quality_for_current_cluster > main_quality:
                        self.medoids = medoids_with_max_quality_for_current_cluster
                        self.node_cluster = dict_of_clusters_with_max_quality_for_current_cluster
                        self.cluster_nodes = dict_of_reversed_clusters_with_max_quality_for_current_cluster
                        print('New Step')
                        print('Old Quality = {}'.format(main_quality))
                        main_quality = max_quality_for_current_cluster
                        print('New Quality = {}'.format(main_quality))
                        print('Medoids: {}'.format(self.medoids))
                        print('='*35)
                    else:
                        stable_sequence += 1
                        if stable_sequence > border_of_stable:
                            print(self.cluster_nodes)
                            return self.node_cluster, self.medoids

if __name__ == '__main__':
    # G = nx.Graph()
    # G.add_edge(0, 1, attr_dict={'a': 1000})
    # G.add_edge(1, 2, attr_dict={'a': 1000})
    # G.add_edge(2, 0, attr_dict={'a': 1000})
    #
    # G.add_edge(3, 4, attr_dict={'a': 1000})
    # G.add_edge(4, 5, attr_dict={'a': 1000})
    # G.add_edge(3, 5, attr_dict={'a': 1000})
    #
    # G.add_edge(6, 7, attr_dict={'a': 1000})
    # G.add_edge(7, 8, attr_dict={'a': 1000})
    # G.add_edge(6, 8, attr_dict={'a': 1000})
    #
    # G.add_edge(1, 3, attr_dict={'a': 1000})
    # G.add_edge(3, 6, attr_dict={'a': 1000})
    #
    # clusterizator = KMedoids(graph=G, k=3, max_parameter=1200)
    # clusters, medoids = clusterizator.k_medoids(border_of_stable=6)
    # print(clusters)
    # print(medoids)

    start_time = time.time()
    step1_time = time.time()
    books_graph = nx.read_gpickle('example_data/books_graph_main_connected_part.p')
    step2_time = time.time()
    print(len(books_graph.edges()))
    # clusters = KMedoids(graph=books_graph, k=220, max_parameter=2921242)
    # clusters, medoids = clusters.k_medoids(border_of_stable=440, s=2)
    # print(step1_time - start_time)
    # print(step2_time - start_time)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(medoids)
