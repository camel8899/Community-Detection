import numpy as np
import os
import argparse
import random
import operator
from scipy import linalg
from sklearn import preprocessing
from sklearn.cluster import spectral_clustering,SpectralClustering,AffinityPropagation
from scipy.sparse.csgraph import connected_components
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations

base_path = os.path.dirname(os.path.realpath(__file__))

def read_graph(load_path,verbose):
    if verbose > 0:
        print("Loading graph from {} ...".format(load_path))
    Y = []
    with open(load_path,'r') as g:
        for line in g.readlines():
            if len(line) == 0:
                break
            Y.append(line.strip().split(' '))
    return np.array(Y)

def read_ans(load_path,verbose):
    if verbose > 0:
        print("Loding answer from {} ...".format(load_path))
    ans = []
    with open(load_path,'r') as f:
        for i,line in enumerate(f.readlines()):
            if len(line) == 0:
                break
            if i > 0:
                ans.append(line.strip().split(',')[1])
    return np.array(ans,dtype = 'int')

def main():
    parser = argparse.ArgumentParser(description='Doing classification on non-overlapping graph')
    parser.add_argument('--Nodes',type = int, default = 100, help = 'Number of nodes')
    parser.add_argument('--boundary',nargs = '+',type = int, help = 'Boundary for deciding overlapping')
    parser.add_argument('--Method',type = str, default = 'spectral',choices = ['spectral','aff'], 
        help = 'Methods for detection')
    parser.add_argument('--load',type = str, default = 'Save_MMSBM', help = 'Load path')
    parser.add_argument('--plot',action = 'store_true',help = 'Plot the figure')
    parser.add_argument('--save',type = str, default = 'plot_MMSBM', help = 'Save path for plot')
    parser.add_argument('--verbose',type = int, default = 0, help = 'Verbose mode')
    args = parser.parse_args()

    load_path = os.path.join(base_path,args.load)
    Y = read_graph(load_path+'\\graph_{}.txt'.format(args.Nodes),args.verbose)
    ans = read_ans(load_path+'\\ans_{}.txt'.format(args.Nodes),args.verbose)
    k = max(ans) # number of communities, but may be wrong...
   
    if len(args.boundary) != 2:
        print("There should be 2 numbers for deciding the boundary!")
    else:
        l = min(args.boundary)
        u = max(args.boundary)
        if l < 0 or l > int(k/2) or u < 0 or u > int(k/2):
            print("Wrong range of l or u...")
        else:
            print("Start detecting community with {} communities, lower boundary {} and upper boundary {}".format(k,l,u))
            save_path = os.path.join(base_path,args.save)
            if not os.path.exists(save_path):
                if args.verbose > 0:
                    print("{} not exist, creating...".format(save_path))
                os.makedirs(save_path)

            v_map = {}
            for i,m in enumerate(ans):
                if m <= l or m >= k-l:
                    v_map[str(i)] = 0
                elif m >= u and m <= k-u:
                    v_map[str(i)] = 1
                else:
                    v_map[str(i)] = 0.5
            care = 0
            for i in v_map.values():
                if i != 0.5:
                    care += 1
            print("Number of nodes we care: ",care)

            if args.plot:                
                #plot the ground truth
                plot = nx.Graph()
                for i in range(Y.shape[0]):
                	plot.add_node(str(i))

                for i,v in enumerate(Y):
                    for j,n in enumerate(v):
                        if int(n) == 1:
                            plot.add_edge(str(i),str(j))

                fig1 = plt.figure()
                values = [v_map.get(node) for node in plot.nodes()]

                nx.draw(plot, cmap=plt.get_cmap('jet'), node_color=values,node_size = 10,width = 0.2)
                fig1.savefig(os.path.join(save_path,"gt.png"),dpi = 500) # save as png
                
            cat_gt = [[] for i in range(k+1)]               
            for idx,label in enumerate(ans):
                cat_gt[int(label)].append(idx)
            #Calculate the conectivity of each cluster
            connect_gt = [0 for i in range(k+1)]
            for idx,cluster in enumerate(cat_gt):
                if len(cluster) > 1:
                    for node in cluster:
                        others = list(cluster)
                        others.remove(node)
                        for other_node in others: 
                            if int(Y[node][other_node]) == 1:
                                connect_gt[idx] += (1/len(cluster)/(len(cluster)-1))
            if args.verbose > 0:
                print("The connectivity of each community of ground truth is :\n",connect_gt)
            
            if args.Method == 'spectral':
                spec = SpectralClustering(affinity = 'precomputed',n_clusters = k+1)
                labels = spec.fit_predict(Y)
                
                cat = [[] for i in range(k+1)]               
                for idx,label in enumerate(labels):
                    cat[int(label)].append(idx)
                #Calculate the conectivity of each cluster
                connect = [0 for i in range(k+1)]
                for idx,cluster in enumerate(cat):
                    if len(cluster) > 1:
                        for node in cluster:
                            others = list(cluster)
                            others.remove(node)
                            for other_node in others: 
                                if int(Y[node][other_node]) == 1:
                                    connect[idx] += (1/len(cluster)/(len(cluster)-1))
                if args.verbose > 0:
                    print("The connectivity of each community is :\n",connect)
                
                '''
                bad_output = [[] for i in range(args.Nodes)]
                for idx,sort in enumerate(range(16)):
                    for node in cat[sort]:
                        if idx < k-(u+l):
                            bad_output[node] = 1
                        else:
                            bad_output[node] = 0
                err_bad = 0
                for idx,result in enumerate(bad_output):
                    if v_map[str(idx)] != 0.5 and v_map[str(idx)] != result:
                        err_bad += 1 
                if args.verbose > 0:
                    print("The output answer of each node by spectral clustering bad output :\n",bad_output)
                print("Error rate of spectral clustering bad output: {}".format(err_bad/care))
                '''
                sort_index = np.argsort(np.array(connect))
                output = [[] for i in range(args.Nodes)]
                for idx,sort in enumerate(sort_index):
                    for node in cat[sort]:
                        if idx < k-(u+l):
                            output[node] = 1
                        else:
                            output[node] = 0
                err = 0
                sorted_v_map = sorted(v_map.items(),key = operator.itemgetter(0))
                sorted_v_map = [v[1] for v in sorted_v_map]
                if args.verbose > 0:
                    print("The output answer of each node by spectral clustering :\n",output)
                    print("The ground truth :\n",sorted_v_map)
                for idx,result in enumerate(output):
                    if v_map[str(idx)] != 0.5 and v_map[str(idx)] != result:
                        err += 1 
                print("Error rate of spectral clustering : {}".format(err/care))

                #view as just overlap and non-overlap
                
                spec_o = SpectralClustering(affinity = 'precomputed',n_clusters = 2)
                labels_o = spec_o.fit_predict(Y)
                err1 = 0
                err2 = 0
                for idx,result in enumerate(labels_o):
                    if v_map[str(idx)] != 0.5 and v_map[str(idx)] != result:
                        err1 += 1
                    elif v_map[str(idx)] != 0.5 and v_map[str(idx)] == result:
                        err2 += 1
                print("Error rate of spectral clustering only fitting 2 components: {}".format(min(err1,err2)/care))
                
                v_spec_map = {}
                for i,v in enumerate(output):
                    v_spec_map[str(i)] = v
                fig2 = plt.figure()
                values_spec = [v_spec_map.get(node) for node in plot.nodes()]
                nx.draw(plot, cmap=plt.get_cmap('jet'), node_color=values_spec,node_size = 10,width = 0.2)
                fig2.savefig(os.path.join(save_path,"Spectral.png"),dpi = 500) # save as png

            if args.Method == 'aff':
                aff = AffinityPropagation(affinity = 'precomputed',verbose = True,max_iter = 3000)
                labels = aff.fit_predict(Y)
                print(labels)
            '''
            err_min = 2
            output = []
            for idx in range(len(p)): #for all permutation
                permu = list(labels) #labels will be 0 ~ k
                for i,v in enumerate(permu):
                    permu[i] = p[idx][v] #make a mapping from labels -> a specific representation of p
                for i,lab in enumerate(permu):
                    if lab <= l or lab >= k-l:
                        permu[i] = 0
                    elif lab >= u and lab <= k-u:
                        permu[i] = 1
                    else:
                        if args.verbose > 0:
                            print("There's a not important node {} which belongs to \
                                {} haha".format(i,lab))
                        mid = int((l+u)/2)
                        if lab <= mid or lab >= k-mid:
                            permu[i] = 0
                        else:
                            permu[i] = 1

                err = 0
                for i in range(len(v_map)):
                    if v_map[str(i)] != 0.5 and v_map[str(i)] != permu[i]:
                        err += 1
                if args.verbose > 0:
                    print("The " +str(idx)+" permutation has error rate "+ str(err/args.Nodes))
                if err/N < err_min:
                    err_min = err/N
                    output = permu
            print("Minimum error rate " + str(err_min))
            
            v_spec_map = {}
            for i,v in enumerate(output):
            	v_spec_map[str(i)] = v
            fig2 = plt.figure()
            values_spec = [v_spec_map.get(node) for node in plot.nodes()]
            nx.draw(plot, cmap=plt.get_cmap('jet'), node_color=values_spec,node_size = 10,width = 0.2)
            fig2.savefig(os.path.join(save_path,"Spectral.png"),dpi = 500) # save as png
            
                
            val_diff_map = {}
            for i,v in enumerate(Ans):
            	if output[i] == v:
            		val_diff_map[str(i)] = 1
            	else:
            		val_diff_map[str(i)] = 0
            fig3 = plt.figure()
            values_diff = [val_diff_map.get(node, 0.25) for node in plot.nodes()]
            nx.draw(plot, cmap=plt.get_cmap('jet'), node_color=values_diff,node_size = 10,width = 0.2)
            fig3.savefig("Diff_dis.png",dpi = 500) # save as png
            '''
if __name__ == "__main__":
    main()


