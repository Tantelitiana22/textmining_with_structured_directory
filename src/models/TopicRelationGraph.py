from graphviz import Graph
import numpy as np
from scipy.spatial.distance import pdist,squareform
from gensim.matutils import jensen_shannon
import itertools as itt

class LdaNetwork:
    
    def __distance(self,X,function_metrics):
        return squareform(pdist(X,function_metrics))
  
    def __build_edge(self,topic_dist,threshold,function_metrics):
        
        topic_distance=self.__distance(topic_dist,function_metrics)
        edges=[ (i+1,j+1,{"weight":topic_distance[i,j]}) for i,j in itt.combinations(range(topic_dist.shape[0]),2)]
        
        if threshold<0 or threshold>1:
            print("Warning: threshold most be between 0 and 1")
            return edges
        elif threshold!=0:
            edges2=[e for e in edges if e[2]["weight"]>threshold]
            return edges2
        elif threshold==0:
            return edges
        
    def __build_node(self,topic_dist):
        return ([k for k in range(1,topic_dist.shape[0]+1)])
    
    def GraphViz(self,topic_dist,threshold=0,function_metrics=lambda u,v: jensen_shannon(u,v)):
        
            edges=self.__build_edge(topic_dist=topic_dist,threshold=threshold,function_metrics=function_metrics)
            nodes=self.__build_node(topic_dist)
            
            u=Graph('finite_state_machine', filename='unix.gv')
            u.attr(size='10,5',rankdir='LR')
            u.node_attr.update(color='grey:lightblue', style='filled',shape='doublecircle',size='8,8')


            for k in nodes:
                u.node(name=str(k),label="topic°"+str(k))


            for e in edges:
                u.edge(str(e[0]), str(e[1]), label=str(np.round(e[2]["weight"],2)),penwidth='2.0')

            return u