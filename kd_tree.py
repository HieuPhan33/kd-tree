from copy import deepcopy
import numpy as np
import time
def permute(arr):
    yield arr
    cur = arr
    for _ in range(len(arr)-1):
        tmp = [0]*len(arr)
        tmp[-1] = cur[0]
        tmp[:-1] = cur[1:]
        yield tmp
        cur = tmp

class Node:
    def __init__(self,point):
        self.point = np.array(point)
        self.left = None
        self.right = None

class kdtree:
    def __init__(self,points):
        self.root = None
        self.d = len(points[0])
        dim = list(range(self.d))
        indices = []
        points = np.unique(points,axis=0)
        for permu_dim in permute(dim):
            permu_id = [a for a,b in sorted(enumerate(points),key = lambda x: [x[1][i] for i in permu_dim])]
            indices.append(permu_id)
        self.root = self.constructTree(points,indices,0,len(points)-1)

    def cmp_points(self,pointA,pointB,id):
        for _ in range(len(pointA)):
            if pointA[id] < pointB[id]:
                return -1
            elif pointA[id] > pointB[id]:
                return 1
            id = (id+1) % self.d
        return 0

    def constructTree(self,points,list_indices,start,end,depth=0):
        if start > end:
            return
        if start == end:
            pnt = points[list_indices[0][start]]
            return Node(pnt)

        mid = (start+end)//2
        med = points[list_indices[0][mid]]
        node = Node(med)
        # print("before depth ", depth,"    ",node.point)
        tmp = deepcopy(list_indices[0])

        for copy_to,indices in enumerate(list_indices[1:]):
            x = 0
            prev_indices = list_indices[copy_to] # Copy indices to prev_indices
            start_1 = start
            start_2 = mid + 1
            for pnt_id in indices[start:end+1]:
                pnt = points[pnt_id]
                cmp = self.cmp_points(pnt,med,(depth%self.d))
                if cmp == 0:
                    continue
                elif cmp == -1:
                    #print(med," vs ",pnt)
                    prev_indices[start_1] = pnt_id
                    start_1 += 1
                else:
                    x+= 1
                    prev_indices[start_2] = pnt_id
                    start_2 += 1
        list_indices[-1] = tmp

        node.left = self.constructTree(points,deepcopy(list_indices),start,mid-1,depth+1)
        node.right = self.constructTree(points,deepcopy(list_indices),mid+1,end,depth+1)
        # print("after depth ",depth, "    ", node.point, " L ",node.left.point," R ",node.right.point)
        return node

    def inorder_traverse(self,node,results):
        if node == None:
            return
        self.inorder_traverse(node.left,results)
        results.append(node.point)
        self.inorder_traverse(node.right,results)

    def close_point(self,p,pointA,pointB):
        if pointA is None and pointB is not None:
            return pointB

        if pointB is None and pointA is not None:
            return pointA
        if pointA is None and pointB is None:
            return None
        distA = np.linalg.norm(p - pointA)
        distB = np.linalg.norm(p - pointB)
        if distA < distB:
            return pointA
        if distB < distA:
            return pointB
        return pointA

    def nearest_neighbor(self,Q):
        return self.nearest_neighbor_helper(self.root,np.array(Q))

    def nearest_neighbor_helper(self,node,Q,depth = 0):
        if node is None:
            return None

        d = depth % self.d
        cmp = self.cmp_points(Q,node.point,d)
        best = 0
        if cmp == 0 :
            return node
        elif cmp == -1:
            searching = node.left
            opposite = node.right
        else:
            searching = node.right
            opposite = node.left

        best = self.nearest_neighbor_helper(searching, Q, depth + 1)
        best = self.close_point(Q, best, node.point)
        dist = np.linalg.norm(Q - best)
        # If the radius of the nearest neighbor still cuts the splitting plane
        # We need to search in the opposite side
        if dist > np.abs(Q[d] - node.point[d]):
            opposite_best = self.nearest_neighbor_helper(opposite, Q, depth + 1)
            best = self.close_point(Q, best, opposite_best)
        return best

    def range_query(self,intervals):
        results = []
        self.range_query_helper(intervals,self.root,results)
        return results

    def range_query_helper(self,intervals,node,results,depth=0):
        d = depth % self.d
        if intervals[d][0] > intervals[d][1]:
            assert "Start must be greater than End at interval with index " + str(d)

        if node is None:
            return

        if belong_to(node.point,intervals):
            results.append(node.point)

        if intervals[d][1] < node.point[d]:
            # Traverse left
            self.range_query_helper(intervals,node.left,results,depth+1)

        elif intervals[d][0] > node.point[d]:
            # Traverse right
            self.range_query_helper(intervals,node.right,results,depth+1)

        else:# Overlapping both sides
            self.range_query_helper(intervals, node.left, results, depth + 1)
            self.range_query_helper(intervals, node.right, results, depth + 1)

def range_query(points,intervals):
    results = []
    for point in points:
        if belong_to(point,intervals):
            results.append(point)
    return results

def belong_to(point,intervals):
    for d,interval in enumerate(intervals):
        if point[d] < interval[0] or point[d] > interval[1]:
            return False
    return True

def main():
    np.random.seed(0)
    points = np.random.rand(10000,2)
    intervals = [[0.1, 0.2], [0.5, 0.9]]
    # points = [
    #     [1,2],
    #     [3,3],
    #     [4,5],
    #     [2,10],
    #     [3,8],
    #     [4,9],
    # ]
    # intervals = [[2,5],[3,9]]

    s = time.time()
    results = range_query(points,intervals)
    e = time.time()
    print("Query without kd-tree")
    print("Querying time ",(e-s)," seconds")
    print("Results ",len(results)," points")

    s = time.time()
    tree = kdtree(points)
    e = time.time()
    print("Time building kd-tree :", (e-s)," seconds")

    # Q = [0.1,0.4,0.5,0.6]
    # nearest = tree.nearest_neighbor(Q)
    # print("Nearest point to ",Q," is ",nearest)

    s = time.time()
    results = tree.range_query(intervals)
    e = time.time()
    print("Query kd-tree")
    print("Time querying ", (e-s), " seconds")
    print(len(results)," points")



if __name__ == '__main__':
    main()