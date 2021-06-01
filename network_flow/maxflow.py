from random import randint, sample
from collections import deque
import numpy as np
from graphviz import Digraph

class maxflow:
    def __init__(self, n):
        # 可視化のために作っているのでnが小さいものだけとする
        if n > 10:
            raise ValueError("too large n")
        # 小さすぎてもだめ
        if n < 3:
            raise ValueError("too small n")
        # nodeの数
        self.n = n
        # edgeの数
        self.m = 0
        # 隣接行列(ただし各要素は容量)
        # 隣接リストにすると計算量が改善できるかもしれないけどめんどくさいので
        # u->vに辺があり、その容量がwの時self.capacity[u][v] == w
        self.capacity = np.array([[0 for _ in range(n)] for _ in range(n)])
        # 隣接行列(ただし各要素は流量)
        # u->vにw流れていたらself.flow[u][v] == w
        self.flow = np.array([[0 for _ in range(n)] for _ in range(n)])
        # 隣接行列(ただし各要素は残余)
        self.residual = np.array([[0 for _ in range(n)] for _ in range(n)])

    def add_edge(self, u, v, w):
        """
        u->vに容量wの辺を加える
        """
        if self.n <= u or self.n <= v:
            raise ValueError
        self.capacity[u][v] = w

    def add_random_edge(self, m, max_capacity=100):
        """
        ランダム(?)に最大max_capacityの容量の辺をm本持つようなネットワークを構成する
        """
        if m < self.n-1:
            raise ValueError("too small m")
        if m > self.n*(self.n-1)//2:
            raise ValueError("too large m")
        if max_capacity < 1:
            raise ValueError("too small max_capacity")
        if max_capacity >= 200:
            raise ValueError("too large max_capacity")

        # 後ろのノードから順に繋いでいく
        for v in reversed(range(1, self.n)):
            lower, upper = max(1, m-self.m-(v-1)*v//2), m-self.m-(v-1)
            #print("lower", lower, "upper", upper)
            in_cnt = randint(lower, upper)
            #print("in_cnt", in_cnt)
            self.m += in_cnt
            in_list = sample(range(v), k=in_cnt)
            # 全ての辺が確実にs->tのパスに入るように調整する
            if np.sum(self.capacity[v-1]) == 0 and v-1 not in in_list:
                in_list[randint(0, in_cnt-1)] = v-1
            for u in in_list:
                self.capacity[u][v] = randint(1, max_capacity)
        

        

    def update_residual(self):
        """
        現在のself.capacityとself.flowから残余グラフを更新する
        """
        for u in range(self.n):
            for v in range(self.n):
                if self.capacity[u][v]:
                    self.residual[u][v] = self.capacity[u][v] - self.flow[u][v]
                    self.residual[v][u] = self.flow[u][v]

    def find_path(self):
        """
        bfsで残余グラフにおいて始点から終点に到るpathを見つける。
        見つかったら順にnode番号を入れたlistを、見つからなければNoneを返す。
        """
        q = deque()
        q.append([0])
        while(len(q)):
            tmp_path = q.popleft()
            u = tmp_path[-1]
            # 今見てるpathが終点に達していたらそれを返す
            if u == self.n-1:
                return tmp_path
            # そうでなければ訪れていない頂点へ行く
            for v in range(self.n):
                if v in tmp_path:
                    continue
                if not self.residual[u][v]:
                    continue
                q.append(tmp_path+[v])

        return None

    def update_flow(self, path):
        """
        pathからflowを更新していく
        """
        if path is None:
            raise ValueError("path is None")
        delta = 500
        # 変更分
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            delta = min(delta, self.residual[u][v])
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if self.capacity[u][v]:
                self.flow[u][v] += delta
            if self.capacity[v][u]:
                self.flow[v][u] -= delta

    def sum_f(self):
        return np.sum(self.flow.T[-1])

    def solve(self):
        """
        増加路法
        """
        for i in range(200*self.n):
            print(f"\rstep {i}", end="")
            self.update_residual()
            path = self.find_path()
            if path is None:
                # 求めた最大流の値と各辺に流れるフローを返す。
                return self.sum_f(), self.flow
            self.update_flow(path)
        return None

class visualize:
    def __init__(self, mf: maxflow):
        # エッジを張った状態のmaxflowを引数にして生成。
        self.maxflow = mf
        self.n = mf.n
        self.m = mf.m
        self.p = None

    def show_graph(self, graph_type, path=None):
        """
        graph_typeの有向グラフを表示する。
        pathに指定されたlistに含まれる辺(u, v)は赤にする。
        """
        # graphviz用のオブジェクト生成
        graph = Digraph(format='svg')

        #ノード追加
        for i in range(self.n):
            graph.node(str(i))

        #エッジ追加
        if graph_type == "capacity":
            g = self.maxflow.capacity
        elif graph_type == "residual":
            g = self.maxflow.residual
        elif graph_type == "flow":
            g = self.maxflow.flow
        else:
            raise ValueError("graph type should be 'capacity', 'residual', or 'flow")

        for i in range(self.n):
            for j in range(self.n):
                if graph_type == "flow":
                    # flowを表示する時はcapacityも同時に表示したい。
                    if g[i][j] == 0 and self.maxflow.capacity[i][j] == 0:
                        continue
                    if path is not None and (i, j) in path:
                        graph.edge(str(i), str(j), label=f"{g[i][j]}/{self.maxflow.capacity[i][j]}", color="red")
                    else:
                        graph.edge(str(i), str(j), label=f"{g[i][j]}/{self.maxflow.capacity[i][j]}")
                elif g[i][j] == 0:
                    continue
                elif path is not None and (i, j) in path:
                    graph.edge(str(i), str(j), label=str(g[i][j]), color="red")
                else:
                    graph.edge(str(i), str(j), label=str(g[i][j]))
        return graph

    def update_residual(self):
        """
        残余ネットワークを更新して表示する。
        """
        self.maxflow.update_residual()
        return self.show_graph(graph_type="residual")

    def show_path(self):
        """
        pathを見つけて表示する。
        """
        self.p = self.maxflow.find_path()
        if self.p is None:
            print("No path was found")
            return None

        path = [(self.p[i], self.p[i+1]) for i in range(len(self.p)-1)]
        return self.show_graph(graph_type="residual", path=path)

    def update_flow(self):
        """
        flowを更新して表示する。
        """
        if self.p is None:
            print("No update")
            return
        self.maxflow.update_flow(self.p)
        path = [(self.p[i], self.p[i+1]) for i in range(len(self.p)-1)]
        return self.show_graph(graph_type="flow", path=path)



