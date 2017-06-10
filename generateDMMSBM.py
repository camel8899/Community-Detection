import numpy as np
import os
import argparse

base_path = os.path.dirname(os.path.realpath(__file__))

def outputMat(Y,save_path,verbose):
	if verbose > 0:
		print("Saving graph to {} ...".format(save_path))
	with open(os.path.join(save_path,'graph_adj_{}.txt'.format(Y.shape[0])), 'w') as text:
		for i in range(Y.shape[0]):
			for j in range(Y.shape[1]):
				text.write(str(int(Y[i, j])) + " ")
			text.write("\n")

	with open(os.path.join(save_path,'graph_edge_{}.txt'.format(Y.shape[0])), 'w') as text:
		for row in Y:
			edge = np.argwhere(row == 1)
			edge = edge.reshape(edge.shape[0],)
			print(edge)
			for neighbor in edge :
				text.write(str(neighbor)+" ")
			text.write('\n')

def outputAns(mbsp, save_path, k, verbose):
	if verbose > 0:
		print("Saving ans to {} ...".format(save_path))
	with open(os.path.join(save_path,'ans_{}.txt'.format(mbsp.shape[1])), 'w') as text:
		text.write("NodeID,Community\n")
		for i in range(mbsp.shape[1]):
			text.write("{},{}\n".format(i+1,int(mbsp[0,i]*k)))
        
def membership(N,k,verbose):
	mbspArr = np.array( [ i for i in range(k+1)] )
	if verbose > 0:
		print("Generating membership vector with {} nodes and {} disjoint communities".format(N,k))

	mbsp = np.array([mbspArr[np.random.randint(k+1)] for i in range(N)])
	mbsp = np.vstack((mbsp,k-mbsp))/k
	if verbose > 0:
		print("The membership for each node: \n{}".format(mbsp.tolist()))
	return mbsp
    
def createGraph(mbsp, B,verbose):
    length = mbsp.shape[1]
    Y = np.zeros((length, length))
    if verbose > 0:
    	print("Creating graph...")
    
    for i in range(length):
        for j in range(i + 1, length):
            temp = mbsp[:,i] @ B @ mbsp[:,j]
            Y[i,j] = np.random.binomial(1,temp,1)
    
    for i in range(length):
        for j in range(0, i):
            Y[i, j] = Y[j, i]
    if verbose > 1:
    	print("Connectivity matrix Y: ")
    	print(Y.tolist())
    return Y

def main():

    parser = argparse.ArgumentParser(description='Generate discrete MMSBM graph')
    parser.add_argument('--Nodes',type = int, default = 100, help = 'Number of nodes')
    parser.add_argument('--C',type = int, default = 15, help = 'Number of communities')
    parser.add_argument('--B',nargs = '+', type = float ,default = [0.5,0.1],help = 'p and q in the connection matrix')
    parser.add_argument('--save',type = str, default = 'Save_MMSBM',help = 'The directory to save the graph and ans')
    parser.add_argument('--verbose',type = int, default = 0, help = 'Verbose mode')
    args = parser.parse_args()
    save_path = os.path.join(base_path,args.save)
    if not os.path.exists(save_path):
    	if args.verbose > 0:
    		print("{} not exist, creating...".format(save_path))
    	os.makedirs(save_path)

    p = args.B[0]
    q = args.B[1]
    if p > 1 or q > 1:
    	print("A probability measure can't be greater than 1! ")
    else:
    	B = np.array([[p, q], [q, p]])
    	mbsp = membership(args.Nodes, args.C,args.verbose)
    	Y = createGraph(mbsp, B, args.verbose)
    	outputMat(Y,save_path,args.verbose)
    	outputAns(mbsp, save_path, args.C, args.verbose)

if __name__ == "__main__":
    main()

