import numpy as np
from sklearn.metrics import average_precision_score

def map_nested_fn(fn):
    """
    Recursively apply fn to the key-value pairs of a nested dict.
    """
    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }
    return map_fn

def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, bu_i * a_j + bu_j

def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
        from https://github.com/rampasek/GraphGPS/blob/main/graphgps/metrics_ogb.py#L31
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)

def pq_walk_sequences(adj, p=1, q=1, walk_length=10, num_walks=5):
    """
    Node2Vec의 p-q walk를 기반으로 walk sequences를 생성.
    """
    num_nodes = adj.shape[0]
    walk_sequences = []

    for start_node in range(num_nodes):
        for _ in range(num_walks):
            current_node = start_node
            prev_node = -1
            walk = [current_node]  # 시작 노드를 포함한 walk
            for _ in range(walk_length - 1):
                neighbors = np.where(adj[current_node] > 0)[0]
                if len(neighbors) == 0:
                    break  # 이동 불가
                
                # p-q 기반 확률 계산
                probabilities = []
                for neighbor in neighbors:
                    if neighbor == prev_node:
                        probabilities.append(1 / p)  # 이전 노드로 되돌아가는 확률
                    elif prev_node != -1 and adj[prev_node, neighbor] > 0:
                        probabilities.append(1)  # 이전 노드와 연결된 노드
                    else:
                        probabilities.append(1 / q)  # 다른 노드로 이동할 확률
                
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()  # 확률 정규화
                
                # 다음 노드 선택
                next_node = np.random.choice(neighbors, p=probabilities)
                walk.append(next_node)
                prev_node = current_node
                current_node = next_node
            
            walk_sequences.append(walk)
    
    return walk_sequences

def gen_dist_mask_pq_walk(adj, p=1, q=1, walk_length=10, num_walks=5):
    """
    p-q walk 기반 거리 마스크 생성.
    """
    walk_sequences = pq_walk_sequences(adj, p=p, q=q, walk_length=walk_length, num_walks=num_walks)
    dist_mask = np.zeros((adj.shape[0], walk_length, adj.shape[0]), dtype=np.int32)

    for walk in walk_sequences:
        for step, node in enumerate(walk[:-1]):  # 마지막 노드는 다음 이동이 없으므로 제외
            if step == 0:
                dist_mask[node, step, node] += 1
            next_node = walk[step + 1]
            dist_mask[walk[0], step + 1, next_node] +=1  # 해당 스텝에서 이동한 노드 쌍을 기록

    return walk_sequences, dist_mask