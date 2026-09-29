import networkx as nx
import numpy as np

def calculate_node_priorities(G, N, gate_node=0):
    """
    Calculate node priorities using various NetworkX centrality measures.
    Returns multiple priority lists that can be used for different allocation strategies.
    
    Parameters:
    - G: NetworkX graph
    - N: Number of nodes
    - gate_node: Gate node index (default: 0)
    
    Returns:
    - dict: Dictionary containing different priority lists
    """
    
    priorities = {}
    
    # 1. Distance-based priority (farther from gate = higher priority for loading)
    shortest_distances = nx.single_source_shortest_path_length(G, gate_node)
    distance_priority = sorted(range(1, N), key=lambda x: shortest_distances.get(x, float('inf')), reverse=True)
    priorities['distance_desc'] = distance_priority  # 게이트에서 먼 순서 (loading용)
    priorities['distance_asc'] = distance_priority[::-1]  # 게이트에서 가까운 순서 (unloading용)
    
    # 2. Betweenness centrality (중요한 경로상의 노드들)
    betweenness = nx.betweenness_centrality(G)
    betweenness_priority = sorted(range(1, N), key=lambda x: betweenness[x], reverse=True)
    priorities['betweenness_desc'] = betweenness_priority  # 높은 betweenness 순
    priorities['betweenness_asc'] = betweenness_priority[::-1]  # 낮은 betweenness 순 (덜 중요한 노드부터)
    
    # 3. Closeness centrality (전체적으로 중심에 있는 노드들)
    closeness = nx.closeness_centrality(G)
    closeness_priority = sorted(range(1, N), key=lambda x: closeness[x], reverse=True)
    priorities['closeness_desc'] = closeness_priority
    priorities['closeness_asc'] = closeness_priority[::-1]
    
    # 4. Degree centrality (연결이 많은 노드들)
    degree = dict(G.degree())
    degree_priority = sorted(range(1, N), key=lambda x: degree[x], reverse=True)
    priorities['degree_desc'] = degree_priority
    priorities['degree_asc'] = degree_priority[::-1]
    
    # 5. Eigenvector centrality (영향력있는 노드들과 연결된 노드들)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        eigenvector_priority = sorted(range(1, N), key=lambda x: eigenvector[x], reverse=True)
        priorities['eigenvector_desc'] = eigenvector_priority
        priorities['eigenvector_asc'] = eigenvector_priority[::-1]
    except:
        # Eigenvector centrality가 수렴하지 않는 경우 degree로 대체
        priorities['eigenvector_desc'] = degree_priority
        priorities['eigenvector_asc'] = degree_priority[::-1]
    
    # 6. Custom composite priority (여러 지표 조합)
    # 거리 + betweenness 역순 조합 (먼데 + 덜 중요한 경로)
    composite_scores = {}
    for node in range(1, N):
        distance_score = shortest_distances.get(node, 0) / max(shortest_distances.values()) if shortest_distances else 0
        betweenness_score = 1 - betweenness[node]  # 역순으로 (낮은 betweenness가 좋음)
        composite_scores[node] = 0.7 * distance_score + 0.3 * betweenness_score
    
    composite_priority = sorted(range(1, N), key=lambda x: composite_scores[x], reverse=True)
    priorities['composite'] = composite_priority
    
    # 7. Strategic loading priority (늦게 나갈 것들을 깊숙이)
    # 이것은 수요 정보가 필요하므로 별도 함수로 분리 예정
    
    return priorities

def get_strategic_priority_for_demands(priorities, K, current_port):
    """
    Get strategic node priority based on demand destinations.
    Later destinations should go to farther nodes from gate.
    
    Parameters:
    - priorities: Priority dictionary from calculate_node_priorities
    - K: Demand list
    - current_port: Current port index
    
    Returns:
    - list: Nodes ordered by strategic priority for current port
    """
    
    # 현재 항구에서 출발하는 수요들의 목적지 분석
    demands_from_port = [(idx, K[idx]) for idx in range(len(K)) 
                        if K[idx][0][0] == current_port]  # K[idx][0][0] = origin
    
    if not demands_from_port:
        return priorities['distance_desc']  # 기본값
    
    # 목적지별로 그룹화
    dest_groups = {}
    for idx, demand_info in demands_from_port:
        dest = demand_info[0][1]  # destination
        qty = demand_info[1]      # quantity
        if dest not in dest_groups:
            dest_groups[dest] = []
        dest_groups[dest].append((idx, qty))
    
    # 목적지가 늦을수록 더 먼 노드에 배치
    sorted_destinations = sorted(dest_groups.keys(), reverse=True)
    
    # 거리 기반 우선순위를 목적지별로 분할
    distance_priority = priorities['distance_desc'].copy()
    strategic_priority = []
    
    nodes_per_dest = len(distance_priority) // len(sorted_destinations) if sorted_destinations else 0
    remainder = len(distance_priority) % len(sorted_destinations) if sorted_destinations else 0
    
    start_idx = 0
    for i, dest in enumerate(sorted_destinations):
        # 뒷 목적지일수록 약간 더 많은 노드 할당
        current_allocation = nodes_per_dest + (1 if i < remainder else 0)
        end_idx = start_idx + current_allocation
        strategic_priority.extend(distance_priority[start_idx:end_idx])
        start_idx = end_idx
    
    return strategic_priority

# 알고리즘에 통합할 수 있는 래퍼 함수
def initialize_node_priorities(prob_info):
    """
    Initialize all node priorities for the given problem.
    
    Parameters:
    - prob_info: Problem information dictionary
    
    Returns:
    - dict: All calculated priorities
    """
    N = prob_info['N']
    E = prob_info['E']
    K = prob_info['K']
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(E)
    
    # Calculate all priorities
    priorities = calculate_node_priorities(G, N)
    
    # Add problem-specific information
    priorities['prob_info'] = prob_info
    
    print("Calculated node priorities:")
    for priority_type, priority_list in priorities.items():
        if priority_type != 'prob_info':
            print(f"  {priority_type}: {priority_list[:5]}..." if len(priority_list) > 5 else f"  {priority_type}: {priority_list}")
    
    return priorities

# 사용 예시:
if __name__ == "__main__":
    # 테스트용 그래프 생성
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2), (1,3), (3,4), (3,5)])
    N = 6
    
    priorities = calculate_node_priorities(G, N)
    print("Test priorities:", priorities)