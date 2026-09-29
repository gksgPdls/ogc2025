# myalgorithm_enhanced_cluster_pathaware.py
# 네트워크 구조 기반 적응적 가중치 + 미래 하역 고려 + 말단 노드 활용
# Last updated: 2025/07/25

import time
import util
import networkx as nx
from itertools import islice
import numpy as np
import priority
import random

def algorithm(prob_info, timelimit=60):
    """
    네트워크 구조 적응적 알고리즘 + 미래 하역 고려 + 말단 노드 활용
    """

    def analyze_network_topology(graph):
        """네트워크 구조 분석 및 적응적 가중치 설정"""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        avg_degree = sum(d for n, d in graph.degree()) / n_nodes
        edge_connectivity = nx.edge_connectivity(graph)
        bridges = len(list(nx.bridges(graph)))
        density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
        
        # 토폴로지 분류 및 가중치 결정
        if bridges > n_nodes * 0.3 or edge_connectivity <= 1 or avg_degree < 2.5:
            return {'type': 'tree', 'criticality_weight': 20, 'terminal_bonus': -15, 'future_penalty_weight': 18}
        elif edge_connectivity >= 3 and density > 0.1 and avg_degree > 4:
            return {'type': 'complex', 'criticality_weight': 8, 'terminal_bonus': -5, 'future_penalty_weight': 8}
        elif edge_connectivity >= 2 and 2.5 <= avg_degree <= 4.5:
            return {'type': 'grid', 'criticality_weight': 12, 'terminal_bonus': -10, 'future_penalty_weight': 10}
        else:
            return {'type': 'standard', 'criticality_weight': 10, 'terminal_bonus': -8, 'future_penalty_weight': 12}

    def identify_terminal_nodes(graph):
        """말단 노드 식별 (degree <= 2)"""
        return [node for node, degree in graph.degree() if degree <= 2 and node != 0]

    def analyze_future_conflicts(current_port):
        """미래 하역 수요 분석"""
        future_demands = {}
        for k_idx, (origin_dest, quantity) in enumerate(K):
            origin, destination = origin_dest
            if origin == current_port and destination > current_port:
                if destination not in future_demands:
                    future_demands[destination] = 0
                future_demands[destination] += quantity
        return future_demands

    def get_critical_path_nodes(graph, target_nodes, gate_node=0):
        """중요 경로 노드 식별"""
        critical_nodes = {}
        for target in target_nodes:
            try:
                path = nx.shortest_path(graph, gate_node, target)
                for node in path[:-1]:
                    critical_nodes[node] = critical_nodes.get(node, 0) + 1
            except nx.NetworkXNoPath:
                continue
        
        threshold = max(2, len(target_nodes) * 0.1)
        return {node: count for node, count in critical_nodes.items() if count >= threshold}

    def find_adaptive_cluster(start_nodes, target_size, available_nodes, weights, critical_nodes, terminal_nodes):
        """적응적 클러스터 형성"""
        if target_size <= 0:
            return []
        
        # 시작 노드 평가
        start_candidates = []
        for node in start_nodes:
            if node in available_nodes:
                criticality = critical_nodes.get(node, 0)
                is_terminal = node in terminal_nodes
                score = (criticality * weights['criticality_weight'] + 
                        (weights['terminal_bonus'] if is_terminal else 0))
                start_candidates.append((score, node))
        
        start_candidates.sort(key=lambda x: x[0])
        
        # BFS 클러스터 확장
        for score, start_node in start_candidates[:3]:
            cluster = {start_node}
            queue = [start_node]
            
            while queue and len(cluster) < target_size:
                current = queue.pop(0)
                neighbors = []
                
                for neighbor in G.neighbors(current):
                    if neighbor in available_nodes and neighbor not in cluster:
                        n_criticality = critical_nodes.get(neighbor, 0)
                        n_is_terminal = neighbor in terminal_nodes
                        n_score = (n_criticality * weights['criticality_weight'] + 
                                 (weights['terminal_bonus'] if n_is_terminal else 0))
                        neighbors.append((n_score, neighbor))
                
                neighbors.sort(key=lambda x: x[0])
                for _, neighbor in neighbors:
                    if len(cluster) < target_size:
                        cluster.add(neighbor)
                        queue.append(neighbor)
                    else:
                        break
            
            if len(cluster) >= target_size:
                return list(cluster)[:target_size]
        
        # Fallback: 순차 할당
        remaining = [n for n in available_nodes if n not in start_nodes]
        fallback_candidates = []
        for n in remaining:
            criticality = critical_nodes.get(n, 0)
            is_terminal = n in terminal_nodes
            score = (criticality * weights['criticality_weight'] + 
                    (weights['terminal_bonus'] if is_terminal else 0))
            fallback_candidates.append((score, n))
        
        fallback_candidates.sort(key=lambda x: x[0])
        result = list(start_nodes)[:target_size]
        for _, node in fallback_candidates:
            if len(result) < target_size:
                result.append(node)
        
        return result[:target_size]

    def loading_heuristic(p, node_allocations, rehandling_demands, node_priorities):
        print(f'Starting loading phase for port {p}...')

        # 수요 수집
        K_load = {idx: K[idx][1] for idx in range(len(K)) if K[idx][0][0] == p}
        print(f'Demands originated from this port: {K_load}')

        if rehandling_demands:
            print(f'Rehandled demands: {rehandling_demands}')
            for k in rehandling_demands:
                k = int(k)
                K_load[k] = K_load.get(k, 0) + 1

        route_list = []
        last_rehandling_demands = []
        total_loading_demands = sum(K_load.values())

        # 도달 가능한 노드 확인
        reachable_nodes, _ = util.bfs(G, node_allocations)
        available_nodes = util.get_available_nodes(node_allocations)

        if len(available_nodes) < total_loading_demands:
            raise Exception("Not enough available nodes!")

        # Rehandling 로직 (기존과 동일)
        if len(reachable_nodes) < total_loading_demands:
            print(f"Rehandling needed...")
            available_but_not_reachable = [n for n in available_nodes if n not in reachable_nodes]
            priority_order = node_priorities['composite']
            available_but_not_reachable.sort(key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))

            while len(reachable_nodes) < total_loading_demands:
                if not available_but_not_reachable:
                    raise Exception("No more nodes for rehandling!")
                
                n = available_but_not_reachable.pop(0)
                distances, previous_nodes = util.dijkstra(G, node_allocations=None)
                path = util.path_backtracking(previous_nodes, 0, n)
                
                blocking_nodes_on_path = [(i, node_allocations[i]) for i in path[:-1] if node_allocations[i] != -1]
                betweenness_asc = node_priorities['betweenness_asc']
                blocking_nodes_on_path.sort(key=lambda x: betweenness_asc.index(x[0]) if x[0] in betweenness_asc else len(betweenness_asc))

                for i, demand_k in blocking_nodes_on_path:
                    last_rehandling_demands.append(demand_k)
                    route_list.append((path[:path.index(i)+1][::-1], demand_k))
                    node_allocations[i] = -1
                    total_loading_demands += 1

                reachable_nodes, _ = util.bfs(G, node_allocations)

        # 최종 수요 업데이트
        for k in last_rehandling_demands:
            k = int(k)
            K_load[k] = K_load.get(k, 0) + 1

        if total_loading_demands > 0:
            print("=== Applying topology-aware adaptive optimization ===")
            
            # 1. 네트워크 분석 (첫 번째 항구에서만)
            if p == 0:
                global topology_info
                topology_info = analyze_network_topology(G)
                print(f"Network topology: {topology_info['type'].upper()}")
                print(f"Weights - criticality: {topology_info['criticality_weight']}, terminal: {topology_info['terminal_bonus']}")
            
            # 2. 현재 상황에 따른 가중치 조정
            remaining_space_ratio = len(available_nodes) / total_loading_demands
            adaptive_weights = topology_info.copy()
            
            if remaining_space_ratio < 1.2:  # 공간 부족 시
                adaptive_weights['criticality_weight'] *= 1.5
                adaptive_weights['terminal_bonus'] *= 1.3
                print(f"Space-constrained adjustment applied (ratio: {remaining_space_ratio:.2f})")
            
            # 3. 말단 노드 및 중요 노드 식별
            terminal_nodes = identify_terminal_nodes(G)
            critical_nodes = get_critical_path_nodes(G, reachable_nodes, 0)
            future_demands = analyze_future_conflicts(p)
            
            print(f"Terminal nodes: {len(terminal_nodes)}, Critical nodes: {len(critical_nodes)}")
            print(f"Future demands: {len(future_demands)} destination ports")
            
            # 4. 수요별 클러스터 할당
            demands_by_destination = {}
            for k, qty in K_load.items():
                k = int(k)
                destination = K[k][0][1]
                if destination not in demands_by_destination:
                    demands_by_destination[destination] = []
                for _ in range(qty):
                    demands_by_destination[destination].append(k)
            
            # 목적지 역순 정렬 (늦게 나갈 것부터, 다음 항구는 마지막)
            sorted_destinations = sorted(demands_by_destination.keys(), reverse=True)
            next_port = p + 1
            if next_port in sorted_destinations:
                sorted_destinations.remove(next_port)
                sorted_destinations.append(next_port)
            
            print(f"Destination priority: {sorted_destinations}")
            
            # 거리별 노드 분류
            reachable_by_distance = {}
            for n in reachable_nodes:
                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                if distance not in reachable_by_distance:
                    reachable_by_distance[distance] = []
                reachable_by_distance[distance].append(n)
            
            # 각 거리층 내에서 말단/중요도 기준 정렬
            for distance in reachable_by_distance:
                nodes_with_priority = []
                for n in reachable_by_distance[distance]:
                    criticality = critical_nodes.get(n, 0)
                    is_terminal = n in terminal_nodes
                    priority = criticality - (3 if is_terminal else 0)  # 말단 노드 우선
                    nodes_with_priority.append((priority, n))
                nodes_with_priority.sort(key=lambda x: (x[0], x[1]))
                reachable_by_distance[distance] = [n for _, n in nodes_with_priority]
            
            loading_assignments = []
            used_nodes = set()
            
            for dest in sorted_destinations:
                dest_demands = demands_by_destination[dest]
                cluster_size = len(dest_demands)
                is_next_port = (dest == next_port)
                
                print(f"  {'*** NEXT PORT ***' if is_next_port else ''} Destination {dest}: {cluster_size} containers")
                
                # 클러스터 형성
                cluster_nodes = []
                distance_order = sorted(reachable_by_distance.keys()) if is_next_port else sorted(reachable_by_distance.keys(), reverse=True)
                
                for distance in distance_order:
                    available_starts = [n for n in reachable_by_distance[distance] if n not in used_nodes]
                    if available_starts:
                        remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                        
                        if is_next_port:
                            # 다음 항구: 단순 연결 클러스터
                            cluster_nodes = available_starts[:cluster_size]
                            if len(cluster_nodes) < cluster_size:
                                additional = [n for n in remaining_available if n not in cluster_nodes]
                                cluster_nodes.extend(additional[:cluster_size - len(cluster_nodes)])
                        else:
                            # 일반: 적응적 클러스터
                            cluster_nodes = find_adaptive_cluster(
                                available_starts[:3], cluster_size, remaining_available,
                                adaptive_weights, critical_nodes, terminal_nodes
                            )
                        
                        if len(cluster_nodes) >= cluster_size:
                            break
                
                # 할당
                cluster_nodes = cluster_nodes[:cluster_size]
                for i, node in enumerate(cluster_nodes):
                    if i < len(dest_demands) and node not in used_nodes:
                        demand_k = dest_demands[i]
                        loading_assignments.append((node, demand_k))
                        used_nodes.add(node)
                        
                        if i < 3:  # 처음 3개만 로그
                            is_terminal = node in terminal_nodes
                            criticality = critical_nodes.get(node, 0)
                            print(f"    Demand {demand_k} -> Node {node} {'(TERMINAL)' if is_terminal else ''} {'(CRITICAL)' if criticality > 0 else ''}")
            
            # 중복 검증
            assert len(loading_assignments) == total_loading_demands, f"Assignment mismatch: {len(loading_assignments)} vs {total_loading_demands}"
            
            # 품질 평가
            total_terminal = sum(1 for node, _ in loading_assignments if node in terminal_nodes)
            total_critical = sum(1 for node, _ in loading_assignments if critical_nodes.get(node, 0) > 0)
            
            print(f"Final quality - Terminal: {total_terminal}/{len(loading_assignments)} ({total_terminal/len(loading_assignments)*100:.1f}%)")
            print(f"               Critical: {total_critical}/{len(loading_assignments)} ({total_critical/len(loading_assignments)*100:.1f}%)")
            
            # 실제 할당
            distances, previous_nodes = util.dijkstra(G, node_allocations)
            for n, k in loading_assignments:
                node_allocations[n] = k
                path = util.path_backtracking(previous_nodes, 0, n)
                route_list.append((path, k))

        print(f'Loading completed: {len(route_list)} routes')
        return route_list, node_allocations

    def unloading_heuristic(p, node_allocations, node_priorities):
        K_unload = {idx: K[idx][1] for idx in range(len(K)) if K[idx][0][1] == p}
        print(f'Starting unloading phase...')
        print(f'Demands destined to this port: {K_unload}')

        route_list = []
        rehandling_demands = []
        terminal_nodes = identify_terminal_nodes(G)

        # 하역 대상을 거리 순으로 정렬 (말단 노드 우선)
        all_unloading_targets = []
        for k, r in K_unload.items():
            unloading_nodes = [n for n in range(N) if node_allocations[n] == k]
            for n in unloading_nodes:
                if node_allocations[n] != -1:
                    distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                    is_terminal = n in terminal_nodes
                    priority = distance - (0.5 if is_terminal else 0)  # 말단 노드 우선
                    all_unloading_targets.append((priority, n, k))
        
        all_unloading_targets.sort(key=lambda x: x[0])
        
        for priority, n, k in all_unloading_targets:
            if node_allocations[n] == -1:
                continue

            # 최단 경로 찾기
            path_options = []
            for path in shortest_paths[n]:
                blocking_nodes = [i for i in path[:-1] if node_allocations[i] != -1]
                path_options.append((len(blocking_nodes), len(path), blocking_nodes, path))
            
            path_options.sort(key=lambda x: (x[0], x[1]))
            min_blocks, _, blocking_nodes, path = path_options[0]

            if min_blocks == 0:
                # 직접 하역 가능
                route_list.append((path[::-1], k))
                node_allocations[n] = -1
            else:
                # 블로킹 노드 제거 후 하역
                for bn in blocking_nodes:
                    if node_allocations[bn] != -1:
                        unload_route = []
                        for i in path:
                            unload_route.append(i)
                            if bn == i:
                                break
                        route_list.append((unload_route[::-1], node_allocations[bn]))
                        
                        if node_allocations[bn] not in K_unload:
                            rehandling_demands.append(node_allocations[bn])
                        node_allocations[bn] = -1
                
                route_list.append((path[::-1], k))
                node_allocations[n] = -1

        print(f'Unloading completed: {len(route_list)} routes, {len(rehandling_demands)} rehandling')
        return route_list, rehandling_demands

    # 메인 실행부
    N = prob_info['N']
    E = prob_info['E']
    K = prob_info['K']
    P = prob_info['P']

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(E)

    print("Initializing priorities...")
    node_priorities = priority.initialize_node_priorities(prob_info)

    # 최단 경로 생성
    shortest_paths = [[]] 
    for i in range(1, N):
        sp_i = list(islice(nx.shortest_simple_paths(G, 0, i), 5))
        shortest_paths.append(sp_i)

    node_allocations = np.ones(N, dtype=int) * -1
    solution = {p: [] for p in range(P)}
    
    # 전역 변수 초기화
    global topology_info
    topology_info = None

    # 각 항구별 처리
    for p in range(P):
        print(f"Port {p} ==============================")

        if p > 0:
            route_list_unload, rehandling_demands = unloading_heuristic(p, node_allocations, node_priorities)
            solution[p].extend(route_list_unload)
        else:
            rehandling_demands = []

        if p < P-1:
            route_list_load, node_allocations = loading_heuristic(p, node_allocations, rehandling_demands, node_priorities)
            solution[p].extend(route_list_load)

        print()

    # 경로 정렬
    for p in range(P):
        if solution[p]:
            loading_routes = []
            unloading_routes = []
            rehandling_routes = []
            
            for route, demand in solution[p]:
                if len(route) >= 2:
                    distance = len(route) - 1
                    if route[0] == 0:  # Loading
                        loading_routes.append((distance, route, demand))
                    elif route[-1] == 0:  # Unloading
                        unloading_routes.append((distance, route, demand))
                    else:  # Rehandling
                        rehandling_routes.append((distance, route, demand))
            
            loading_routes.sort(key=lambda x: -x[0])    # 먼 것부터
            unloading_routes.sort(key=lambda x: x[0])   # 가까운 것부터
            rehandling_routes.sort(key=lambda x: x[0])  # 가까운 것부터
            
            sorted_routes = []
            sorted_routes.extend([(route, demand) for _, route, demand in unloading_routes])
            sorted_routes.extend([(route, demand) for _, route, demand in rehandling_routes])
            sorted_routes.extend([(route, demand) for _, route, demand in loading_routes])
            
            solution[p] = sorted_routes

    return solution

if __name__ == "__main__":
    import json
    import sys
    import jsbeautifier

    def numpy_to_python(obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)  
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    if len(sys.argv) == 4:
        prob_name = sys.argv[1]
        prob_file = sys.argv[2]
        timelimit = int(sys.argv[3])

        with open(prob_file, 'r') as f:
            prob_info = json.load(f)

        try:
            alg_start_time = time.time()
            solution = algorithm(prob_info, timelimit)
            alg_end_time = time.time()

            checked_solution = util.check_feasibility(prob_info, solution)
            checked_solution['time'] = alg_end_time - alg_start_time
            checked_solution['timelimit_exception'] = (alg_end_time - alg_start_time) > timelimit + 2
            checked_solution['exception'] = None
            checked_solution['prob_name'] = prob_name
            checked_solution['prob_file'] = prob_file

            with open('results.json', 'w') as f:
                opts = jsbeautifier.default_options()
                opts.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(checked_solution, default=numpy_to_python), opts))
                print(f'Results are saved as file results.json')
                
            sys.exit(0)

        except Exception as e:
            print(f"Exception: {repr(e)}")
            sys.exit(1)

    else:
        print("Usage: python myalgorithm_enhanced_cluster_pathaware.py <problem_name> <problem_file> <timelimit_in_seconds>")
        sys.exit(2)