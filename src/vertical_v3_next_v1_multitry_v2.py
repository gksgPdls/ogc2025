# myalgorithm_enhanced_cluster_pathaware.py
# 기존 알고리즘 + 고도화된 경로 인식 클러스터 배치 최적화 + 다중 실행 최적화
# Last updated: 2025/07/25

import time
import util
import networkx as nx
from itertools import islice
import numpy as np
import priority
import random
import copy

def algorithm(prob_info, timelimit=60):
    """
    다중 실행을 통한 최적 솔루션 탐색
    """
    
    def run_single_algorithm(prob_info, seed=None, run_id=0):
        """
        단일 알고리즘 실행 (기존 알고리즘 로직)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"--- Run {run_id + 1} (seed: {seed}) ---")
        
        #------------- begin of improved algorithm code --------------#

        def get_critical_path_nodes(graph, target_nodes, gate_node=0):
            """
            중요 경로상의 노드들을 식별 (여러 노드가 공통으로 사용하는 경로)
            """
            critical_nodes = {}
            
            for target in target_nodes:
                try:
                    path = nx.shortest_path(graph, gate_node, target)
                    for node in path[:-1]:  # 목적지 제외
                        if node not in critical_nodes:
                            critical_nodes[node] = 0
                        critical_nodes[node] += 1
                except nx.NetworkXNoPath:
                    continue
            
            # 2개 이상의 경로에서 사용되는 노드들을 중요 노드로 분류
            critical_threshold = max(2, len(target_nodes) * 0.1)  # 최소 2개 또는 10%
            return {node: count for node, count in critical_nodes.items() if count >= critical_threshold}

        def find_path_aware_cluster_dynamic(start_nodes, target_size, available_nodes, graph, critical_nodes, 
                                           criticality_weight=10, connectivity_weight=-30, gate_node=0):
            """
            동적 가중치를 적용한 경로 인식 클러스터 형성
            """
            if target_size <= 0:
                return []
            
            # 여러 시작점에서 클러스터를 시도하고 가장 좋은 것을 선택
            best_cluster = []
            best_score = float('inf')
            
            # 시작 노드들을 중요도에 따라 정렬 (덜 중요한 것부터)
            start_candidates = []
            for node in start_nodes:
                if node in available_nodes:
                    criticality = critical_nodes.get(node, 0)
                    # 게이트로부터의 거리도 고려
                    try:
                        distance = nx.shortest_path_length(graph, gate_node, node)
                    except:
                        distance = float('inf')
                    start_candidates.append((criticality, distance, node))
            
            start_candidates.sort(key=lambda x: (x[0], -x[1]))  # 덜 중요한 것, 먼 것부터
            
            # 여러 시작점에서 클러스터 형성 시도
            for criticality, distance, start_node in start_candidates[:min(3, len(start_candidates))]:
                cluster = build_cluster_from_start_dynamic(start_node, target_size, available_nodes, 
                                                         graph, critical_nodes, criticality_weight, 
                                                         connectivity_weight, gate_node)
                
                if len(cluster) >= target_size:
                    # 클러스터 품질 평가 (동적 가중치 적용)
                    cluster_score = evaluate_cluster_quality_dynamic(cluster, graph, critical_nodes, 
                                                                   available_nodes, criticality_weight, 
                                                                   connectivity_weight, gate_node)
                    
                    if cluster_score < best_score:
                        best_score = cluster_score
                        best_cluster = cluster
            
            return best_cluster[:target_size] if best_cluster else []
        
        def build_cluster_from_start_dynamic(start_node, target_size, available_nodes, graph, critical_nodes, 
                                           criticality_weight, connectivity_weight, gate_node):
            """
            동적 가중치를 적용한 클러스터 구축
            """
            cluster = {start_node}
            queue = [start_node]
            
            # BFS로 클러스터 확장
            while queue and len(cluster) < target_size:
                current = queue.pop(0)
                
                # 이웃 노드들을 품질로 정렬
                neighbor_candidates = []
                for neighbor in graph.neighbors(current):
                    if neighbor in available_nodes and neighbor not in cluster:
                        quality_score = calculate_node_quality_dynamic(neighbor, cluster, available_nodes, 
                                                                     graph, critical_nodes, criticality_weight,
                                                                     connectivity_weight, gate_node)
                        neighbor_candidates.append((quality_score, neighbor))
                
                # 품질이 좋은 순서로 정렬 (점수가 낮을수록 좋음)
                neighbor_candidates.sort(key=lambda x: x[0])
                
                # 좋은 후보들을 클러스터에 추가
                for quality_score, neighbor in neighbor_candidates:
                    if len(cluster) < target_size:
                        cluster.add(neighbor)
                        queue.append(neighbor)
                    else:
                        break
            
            return list(cluster)
        
        def calculate_node_quality_dynamic(node, current_cluster, available_nodes, graph, critical_nodes, 
                                         criticality_weight, connectivity_weight, gate_node):
            """
            동적 가중치를 적용한 노드 품질 계산
            """
            criticality = critical_nodes.get(node, 0)
            
            # 접근성 페널티 계산 (샘플링으로 성능 향상)
            accessibility_penalty = 0
            temp_unavailable = current_cluster | {node}
            
            test_nodes = [n for n in available_nodes if n not in temp_unavailable]
            if len(test_nodes) > 15:  # 15개로 더 제한적으로 샘플링
                test_nodes = random.sample(test_nodes, 15)
            
            for test_node in test_nodes:
                try:
                    temp_graph = graph.copy()
                    temp_graph.remove_nodes_from(temp_unavailable)
                    if not nx.has_path(temp_graph, gate_node, test_node):
                        accessibility_penalty += 1
                except:
                    accessibility_penalty += 1
            
            # 클러스터 연결성 보너스
            connectivity_bonus = 0
            for cluster_node in current_cluster:
                if graph.has_edge(node, cluster_node):
                    connectivity_bonus += 1  # 연결된 노드는 선호
            
            # 게이트로부터의 거리 고려
            try:
                distance_to_gate = nx.shortest_path_length(graph, gate_node, node)
                # 거리 5-8 구간을 선호 (너무 가깝지도, 멀지도 않게)
                optimal_distance = 6
                distance_penalty = abs(distance_to_gate - optimal_distance) * 0.5
            except:
                distance_penalty = 10
            
            # 동적 가중치 적용
            return (criticality * criticality_weight + 
                    accessibility_penalty * 5 + 
                    connectivity_bonus * connectivity_weight + 
                    distance_penalty)
        
        def evaluate_cluster_quality_dynamic(cluster, graph, critical_nodes, available_nodes, 
                                           criticality_weight, connectivity_weight, gate_node):
            """
            동적 가중치를 적용한 클러스터 품질 평가
            """
            # 내부 연결성 점수
            internal_connections = 0
            for i, node1 in enumerate(cluster):
                for node2 in cluster[i+1:]:
                    if graph.has_edge(node1, node2):
                        internal_connections += 1
            
            max_possible_connections = len(cluster) * (len(cluster) - 1) / 2
            connectivity_score = internal_connections / max(1, max_possible_connections)
            
            # 중요 노드 사용 페널티
            critical_penalty = sum(critical_nodes.get(node, 0) for node in cluster)
            
            # 접근성 영향 페널티 (샘플링으로 효율화)
            temp_unavailable = set(cluster)
            remaining_nodes = [n for n in available_nodes if n not in temp_unavailable]
            blocked_count = 0
            
            sample_size = min(15, len(remaining_nodes))  # 더 작은 샘플 크기
            if remaining_nodes:
                test_nodes = random.sample(remaining_nodes, sample_size) if len(remaining_nodes) > sample_size else remaining_nodes
                
                for test_node in test_nodes:
                    try:
                        temp_graph = graph.copy()
                        temp_graph.remove_nodes_from(temp_unavailable)
                        if not nx.has_path(temp_graph, gate_node, test_node):
                            blocked_count += 1
                    except:
                        blocked_count += 1
            
            # 클러스터 응집도 (더 조밀한 클러스터 선호)
            cluster_span = 0
            if len(cluster) > 1:
                distances = []
                for i, node1 in enumerate(cluster):
                    for node2 in cluster[i+1:]:
                        try:
                            dist = nx.shortest_path_length(graph, node1, node2)
                            distances.append(dist)
                        except:
                            distances.append(10)  # 연결되지 않은 경우 높은 페널티
                cluster_span = sum(distances) / len(distances) if distances else 0
            
            # 동적 가중치 적용하여 최종 점수 계산
            return (critical_penalty * criticality_weight + 
                    blocked_count * 20 + 
                    connectivity_score * connectivity_weight + 
                    cluster_span * 5)

        def find_connected_cluster(start_nodes, target_size, available_nodes, graph):
            """
            시작 노드들로부터 BFS를 사용하여 연결된 클러스터를 찾는 함수 (fallback용)
            """
            if target_size <= 0:
                return []
            
            cluster = set()
            queue = list(start_nodes)
            
            # 시작 노드들을 클러스터에 추가
            for node in start_nodes:
                if node in available_nodes:
                    cluster.add(node)
                    if len(cluster) >= target_size:
                        break
            
            # BFS로 인접한 노드들을 클러스터에 추가
            while queue and len(cluster) < target_size:
                current = queue.pop(0)
                
                # 현재 노드의 이웃들을 확인
                for neighbor in graph.neighbors(current):
                    if (neighbor in available_nodes and 
                        neighbor not in cluster and 
                        len(cluster) < target_size):
                        cluster.add(neighbor)
                        queue.append(neighbor)
            
            return list(cluster)

        def loading_heuristic(p, node_allocations, rehandling_demands, node_priorities):
            # All demands that should be loaded at port p
            K_load = {idx: K[idx][1] for idx in range(len(K)) if K[idx][0][0] == p}

            if len(rehandling_demands) > 0:
                for k in rehandling_demands:
                    k = int(k)
                    if k in K_load:
                        K_load[k] += 1
                    else:
                        K_load[k] = 1

            route_list = []
            last_rehandling_demands = []

            total_loading_demands = sum([r for k,r in K_load.items()])

            # Get reachable nodes from the gate
            reachable_nodes, reachable_node_distances = util.bfs(G, node_allocations)
            available_nodes = util.get_available_nodes(node_allocations)

            if len(available_nodes) < total_loading_demands:
                raise Exception("Not enough available nodes to load demand!!! This must not happen!!!")

            if len(reachable_nodes) < total_loading_demands:
                # 기존 rehandling 로직
                available_but_not_reachable = [n for n in available_nodes if n not in reachable_nodes]
                
                priority_order = node_priorities['composite']
                available_but_not_reachable.sort(key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))

                while len(reachable_nodes) < total_loading_demands:
                    if len(available_but_not_reachable) == 0:
                        raise Exception("Not enough available_but_not_reachable nodes to rehandle. This must not happen!!!")
                    
                    n = available_but_not_reachable.pop(0)
                    distances, previous_nodes = util.dijkstra(G, node_allocations=None)
                    path = util.path_backtracking(previous_nodes, 0, n)

                    blocking_nodes_on_path = [(i, node_allocations[i]) for i in path[:-1] if node_allocations[i] != -1]
                    
                    betweenness_asc = node_priorities['betweenness_asc']
                    blocking_nodes_on_path.sort(key=lambda x: betweenness_asc.index(x[0]) if x[0] in betweenness_asc else len(betweenness_asc))

                    for idx, (i, demand_k) in enumerate(blocking_nodes_on_path):
                        last_rehandling_demands.append(demand_k)
                        route_list.append((path[:path.index(i)+1][::-1], demand_k))
                        node_allocations[i] = -1
                        total_loading_demands += 1

                    reachable_nodes, reachable_node_distances = util.bfs(G, node_allocations)

            # Merge the rehandling demands with the loading demands
            for k in last_rehandling_demands:
                k = int(k)
                if k in K_load:
                    K_load[k] += 1
                else:
                    K_load[k] = 1

            if total_loading_demands > 0:
                # IMPROVED: 고도화된 경로 인식 클러스터 배치 + 다음 항구 우선 배치
                
                # 1. 수요를 목적지별로 분류하고 역순 정렬
                demands_by_destination = {}
                for k, qty in K_load.items():
                    k = int(k)
                    destination = K[k][0][1] 
                    if destination not in demands_by_destination:
                        demands_by_destination[destination] = []
                    for _ in range(qty):
                        demands_by_destination[destination].append(k)
                
                # 목적지 역순 정렬 (늦게 나갈 것부터)
                sorted_destinations = sorted(demands_by_destination.keys(), reverse=True)
                
                # 1.5. 다음 항구(p+1) 수요 분리 및 우선 처리 계획
                next_port = p + 1
                next_port_demands = demands_by_destination.get(next_port, [])
                
                # 목적지별 배치 전략 수정: 다음 항구 수요를 가장 마지막에 처리 (가장 가까운 곳에 배치)
                if next_port_demands:
                    # 다음 항구 수요를 제외한 나머지를 먼저 처리
                    priority_destinations = [dest for dest in sorted_destinations if dest != next_port]
                    priority_destinations.append(next_port)  # 다음 항구를 마지막에 추가
                else:
                    priority_destinations = sorted_destinations
                
                # 2. 중요 경로 노드들 식별
                critical_nodes = get_critical_path_nodes(G, reachable_nodes, 0)
                
                # 3. 거리별 노드 분류 및 중요도 고려 정렬
                reachable_nodes_by_distance = {}
                for n in reachable_nodes:
                    distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                    if distance not in reachable_nodes_by_distance:
                        reachable_nodes_by_distance[distance] = []
                    reachable_nodes_by_distance[distance].append(n)
                
                # 각 거리층을 중요도 고려하여 정렬 (덜 중요한 노드 우선)
                for distance in reachable_nodes_by_distance:
                    nodes_with_criticality = []
                    for n in reachable_nodes_by_distance[distance]:
                        criticality = critical_nodes.get(n, 0)
                        nodes_with_criticality.append((criticality, n))
                    nodes_with_criticality.sort(key=lambda x: (x[0], x[1]))  # 중요도 낮은 것, 노드ID 순
                    reachable_nodes_by_distance[distance] = [n for _, n in nodes_with_criticality]
                
                # 4. 동적 우선순위와 함께 경로 인식 클러스터 할당
                loading_assignments = []
                used_nodes = set()
                
                for dest_idx, dest in enumerate(priority_destinations):
                    dest_demands = demands_by_destination[dest]
                    cluster_size = len(dest_demands)
                    
                    # 다음 항구 수요인지 확인
                    is_next_port = (dest == next_port)
                    
                    # 현재 상황에 맞게 중요도 가중치 동적 조정
                    remaining_demands = sum(len(demands_by_destination[d]) for d in priority_destinations[dest_idx:])
                    remaining_space = len([n for n in reachable_nodes if n not in used_nodes])
                    
                    # 공간이 부족할수록 더 보수적으로 (중요 노드 회피)
                    criticality_weight = 15 if remaining_space < remaining_demands * 1.2 else 10
                    # 연결성이 더 중요해짐
                    connectivity_weight = -40 if remaining_space < remaining_demands * 1.5 else -30
                    
                    # 클러스터 형성 전략: 다음 항구는 가까운 곳부터, 나머지는 먼 곳부터
                    if is_next_port:
                        # 다음 항구 수요: 가까운 거리부터 배치 (게이트 근처 우선)
                        distance_order = sorted(reachable_nodes_by_distance.keys())  # 가까운 거리부터
                    else:
                        # 일반 수요: 먼 거리부터 배치
                        distance_order = sorted(reachable_nodes_by_distance.keys(), reverse=True)  # 먼 거리부터
                    
                    # 클러스터 형성을 위한 시작점 찾기
                    cluster_nodes = []
                    
                    for distance in distance_order:
                        available_start_nodes = [n for n in reachable_nodes_by_distance[distance] 
                                               if n not in used_nodes]
                        
                        if available_start_nodes:
                            # 클러스터 형성 방식도 다음 항구에 따라 조정
                            remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                            
                            if is_next_port:
                                # 다음 항구: 간단한 연결된 클러스터 (빠른 접근 우선)
                                cluster_nodes = find_connected_cluster(
                                    available_start_nodes[:1],  # 첫 번째 노드부터 시작
                                    cluster_size, 
                                    remaining_available, 
                                    G
                                )
                                if len(cluster_nodes) >= cluster_size:
                                    break
                            else:
                                # 일반 수요: 고도화된 경로 인식 클러스터 형성
                                cluster_nodes = find_path_aware_cluster_dynamic(
                                    available_start_nodes[:3],  # 상위 3개 후보
                                    cluster_size, 
                                    remaining_available, 
                                    G, 
                                    critical_nodes,
                                    criticality_weight,
                                    connectivity_weight
                                )
                                if len(cluster_nodes) >= cluster_size:
                                    break
                    
                    # 클러스터를 찾지 못한 경우, fallback 방식 사용
                    if len(cluster_nodes) < cluster_size:
                        if is_next_port:
                            # 다음 항구: 가까운 거리부터 순차적으로 할당
                            remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                            # 거리별로 정렬 (가까운 것부터)
                            remaining_with_distance = []
                            for n in remaining_available:
                                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                                remaining_with_distance.append((distance, n))
                            remaining_with_distance.sort(key=lambda x: x[0])  # 가까운 것부터
                            
                            cluster_nodes = [n for _, n in remaining_with_distance[:cluster_size]]
                        else:
                            # 일반 수요: 기존 fallback 방식
                            # Fallback 1: 기본 연결된 클러스터 찾기
                            for distance in sorted(reachable_nodes_by_distance.keys(), reverse=True):
                                available_start_nodes = [n for n in reachable_nodes_by_distance[distance] 
                                                       if n not in used_nodes]
                                if available_start_nodes:
                                    remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                                    cluster_nodes = find_connected_cluster([available_start_nodes[0]], cluster_size, 
                                                                         remaining_available, G)
                                    if len(cluster_nodes) >= cluster_size:
                                        break
                    
                    # 최종 fallback: 중요도 순서대로 순차 할당
                    if len(cluster_nodes) < cluster_size:
                        remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                        
                        if is_next_port:
                            # 다음 항구: 거리 우선 정렬
                            remaining_with_priority = []
                            for n in remaining_available:
                                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                                remaining_with_priority.append((distance, n))
                            remaining_with_priority.sort(key=lambda x: x[0])  # 가까운 것부터
                            cluster_nodes = [n for _, n in remaining_with_priority[:cluster_size]]
                        else:
                            # 일반 수요: 중요도에 따라 정렬
                            remaining_with_criticality = []
                            for n in remaining_available:
                                criticality = critical_nodes.get(n, 0)
                                remaining_with_criticality.append((criticality, n))
                            remaining_with_criticality.sort(key=lambda x: x[0])  # 덜 중요한 것부터
                            cluster_nodes = [n for _, n in remaining_with_criticality[:cluster_size]]
                    
                    # 클러스터 내에서 정렬 (다음 항구는 가까운 순서, 나머지는 기존 방식)
                    cluster_nodes_with_distance = []
                    for n in cluster_nodes:
                        distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                        criticality = critical_nodes.get(n, 0)
                        cluster_nodes_with_distance.append((distance, criticality, n))
                    
                    if is_next_port:
                        # 다음 항구: 가까운 순서 (게이트 출입구 우선)
                        cluster_nodes_with_distance.sort(key=lambda x: (x[0], x[1]))  # 가까운 것부터, 중요도 낮은 것부터
                    else:
                        # 일반 수요: 거리 우선, 중요도는 낮은 것 우선
                        cluster_nodes_with_distance.sort(key=lambda x: (x[0], x[1]))
                    
                    # 할당
                    assigned_count = 0
                    for i, (distance, criticality, node) in enumerate(cluster_nodes_with_distance):
                        if assigned_count < len(dest_demands) and node not in used_nodes:  # 중복 할당 방지
                            demand_k = dest_demands[assigned_count]
                            loading_assignments.append((node, demand_k))
                            used_nodes.add(node)
                            assigned_count += 1
                        elif node in used_nodes:
                            continue
                    
                    # 할당이 부족한 경우 추가 노드 찾기
                    if assigned_count < len(dest_demands):
                        remaining_demands = len(dest_demands) - assigned_count
                        
                        # 사용 가능한 추가 노드 찾기
                        additional_nodes = []
                        for n in reachable_nodes:
                            if n not in used_nodes and len(additional_nodes) < remaining_demands:
                                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                                criticality = critical_nodes.get(n, 0)
                                additional_nodes.append((distance, criticality, n))
                        
                        # 거리와 중요도로 정렬
                        additional_nodes.sort(key=lambda x: (x[0], x[1]))
                        
                        for distance, criticality, node in additional_nodes:
                            if assigned_count < len(dest_demands):
                                demand_k = dest_demands[assigned_count]
                                loading_assignments.append((node, demand_k))
                                used_nodes.add(node)
                                assigned_count += 1

                # 최종 중복 검증
                node_usage = {}
                duplicate_found = False
                for node, demand_k in loading_assignments:
                    if node in node_usage:
                        duplicate_found = True
                    else:
                        node_usage[node] = demand_k
                
                if duplicate_found:
                    # 중복 제거 및 재할당
                    unique_assignments = []
                    used_in_final = set()
                    missing_demands = []
                    
                    for node, demand_k in loading_assignments:
                        if node not in used_in_final:
                            unique_assignments.append((node, demand_k))
                            used_in_final.add(node)
                        else:
                            missing_demands.append(demand_k)
                    
                    # 누락된 수요를 위한 추가 노드 찾기
                    for demand_k in missing_demands:
                        for n in reachable_nodes:
                            if n not in used_in_final:
                                unique_assignments.append((n, demand_k))
                                used_in_final.add(n)
                                break
                    
                    loading_assignments = unique_assignments
                    used_nodes = used_in_final

                assert(len(loading_assignments) == total_loading_demands)

                # 7. 실제 할당 및 경로 생성
                distances, previous_nodes = util.dijkstra(G, node_allocations)

                for n, k in loading_assignments:
                    node_allocations[n] = k
                    path = util.path_backtracking(previous_nodes, 0, n)
                    route_list.append((path, k))

            return route_list, node_allocations

        def unloading_heuristic(p, node_allocations, node_priorities):
            # All demands that should be unloaded at port p
            K_unload = {idx: K[idx][1] for idx in range(len(K)) if K[idx][0][1] == p}

            route_list = []
            rehandling_demands = []

            # 거리 순으로 unloading 처리 (가까운 것부터)
            all_unloading_targets = []
            for k, r in K_unload.items():
                unloading_nodes = [n for n in range(N) if node_allocations[n] == k]
                for n in unloading_nodes:
                    if node_allocations[n] != -1:
                        distance_to_gate = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                        all_unloading_targets.append((distance_to_gate, n, k))
            
            all_unloading_targets.sort(key=lambda x: x[0])  # 가까운 것부터
            
            for distance_to_gate, n, k in all_unloading_targets:
                    if node_allocations[n] == -1:
                        continue

                    num_blocking_nodes = []

                    for path in shortest_paths[n]:
                        blocking_nodes = [i for i in path[:-1] if node_allocations[i] != -1]
                        num_blocking_nodes.append((len(blocking_nodes), len(path), blocking_nodes, path))
                        
                    num_blocking_nodes.sort(key=lambda x: (x[0], x[1]))
                    min_blocking_nodes = num_blocking_nodes[0]

                    if min_blocking_nodes[0] == 0:
                        path = min_blocking_nodes[3]
                        route_list.append((path[::-1], k))
                        node_allocations[n] = -1
                    else:
                        blocking_nodes = min_blocking_nodes[2]
                        path = min_blocking_nodes[3]
                        
                        blocking_nodes_with_distance = []
                        for bn in blocking_nodes:
                            if node_allocations[bn] != -1:
                                bn_distance = len(shortest_paths[bn][0]) - 1 if shortest_paths[bn] else float('inf')
                                blocking_nodes_with_distance.append((bn_distance, bn))
                        
                        blocking_nodes_with_distance.sort(key=lambda x: x[0])
                        
                        for bn_distance, bn in blocking_nodes_with_distance:
                            if node_allocations[bn] == -1:
                                continue
                                
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

            return route_list, rehandling_demands

        # Get parameters from the prob_info dictionary
        N = prob_info['N']
        E = prob_info['E']
        E = set([(u,v) for (u,v) in E])
        K = prob_info['K']
        P = prob_info['P']
        F = prob_info['F']

        # Create a graph for the problem
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(E)

        # 노드 우선순위 초기화
        node_priorities = priority.initialize_node_priorities(prob_info)

        # Shortest distances from the gate
        shortest_distances = np.zeros(N, dtype=int)

        # Generate k-shortest paths from the gate to all nodes
        max_num_paths = 5
        shortest_paths = [[]] 

        for i in range(1, N):
            sp_i = list(islice(nx.shortest_simple_paths(G, 0, i), max_num_paths))
            shortest_paths.append(sp_i)
            shortest_distances[i] = len(sp_i[0]) - 1

        # Convert the shortest paths to a numpy array using indicator vectors
        shortest_paths_array = [[]] 

        for i in range(1, N):
            sp_i_array = []
            for p in shortest_paths[i]:
                path_array = np.zeros(N, dtype=int)
                path_array[p[:-1]] = 1
                sp_i_array.append(path_array)
            shortest_paths_array.append(sp_i_array)

        # Current status of nodes
        node_allocations = np.ones(N, dtype=int) * -1

        # Solution dictionary
        solution = {p: [] for p in range(P)}

        # Loop over all ports and apply the loading and unloading heuristics
        for p in range(P):
            if p > 0:
                # Unloading heuristic
                route_list_unload, rehandling_demands = unloading_heuristic(p, node_allocations, node_priorities)
                solution[p].extend(route_list_unload)
            else:
                rehandling_demands = []

            if p < P-1:
                # Loading heuristic with enhanced path-aware cluster placement
                route_list_load, node_allocations = loading_heuristic(p, node_allocations, rehandling_demands, node_priorities)
                solution[p].extend(route_list_load)

        # Path 정렬 - 충돌 최소화를 위한 고도화된 정렬
        for p in range(P):
            if solution[p]:
                loading_routes = []
                unloading_routes = []
                rehandling_routes = []
                
                for route, demand in solution[p]:
                    if len(route) >= 2:
                        if route[0] == 0:  # Loading route
                            distance = len(route) - 1
                            # 경로 복잡도도 고려 (경로상 노드 수)
                            complexity = len(set(route))
                            loading_routes.append((distance, complexity, route, demand))
                        elif route[-1] == 0:  # Unloading route
                            distance = len(route) - 1
                            complexity = len(set(route))
                            unloading_routes.append((distance, complexity, route, demand))
                        else:  # Rehandling route
                            distance = len(route) - 1
                            complexity = len(set(route))
                            rehandling_routes.append((distance, complexity, route, demand))
                
                # 각 타입별로 거리 순 정렬 (복잡도 고려)
                loading_routes.sort(key=lambda x: (-x[0], x[1]))    # Loading: 먼 것부터, 단순한 것부터
                unloading_routes.sort(key=lambda x: (x[0], x[1]))   # Unloading: 가까운 것부터, 단순한 것부터
                rehandling_routes.sort(key=lambda x: (x[0], x[1]))  # Rehandling: 가까운 것부터, 단순한 것부터
                
                # 순서 유지: Unloading → Rehandling → Loading
                sorted_routes = []
                sorted_routes.extend([(route, demand) for _, _, route, demand in unloading_routes])
                sorted_routes.extend([(route, demand) for _, _, route, demand in rehandling_routes])
                sorted_routes.extend([(route, demand) for _, _, route, demand in loading_routes])
                
                solution[p] = sorted_routes

        #------------- end of improved algorithm code --------------#

        return solution

    # 메인 다중 실행 로직
    start_time = time.time()
    
    # 적응적 실행 횟수 결정 (더 보수적으로 조정)
    if timelimit <= 30:
        max_runs = 2
        min_time_per_run = 12  # 30초에서 2회면 각각 최소 12초
    elif timelimit <= 60:
        max_runs = 4
        min_time_per_run = 12  # 60초에서 4회면 각각 최소 12초
    elif timelimit <= 120:
        max_runs = 6
        min_time_per_run = 15  # 120초에서 6회면 각각 최소 15초
    else:
        max_runs = 10
        min_time_per_run = 15  # 더 긴 시간에서는 각각 최소 15초
    
    print(f"Starting multi-run optimization with max {max_runs} runs (timelimit: {timelimit}s)")
    
    best_solution = None
    best_cost = float('inf')
    best_run_info = {}
    all_costs = []
    
    # 고도화된 시드 다양화 전략
    base_seed = int(time.time()) % 10000
    
    # 다양한 시드 생성 전략
    seeds = []
    
    # 1. 기본 시퀀스 시드
    for i in range(max_runs // 2 + 1):
        seeds.append(base_seed + i * 1371)  # 소수 간격 사용
    
    # 2. 역순 시퀀스 시드 (다른 탐색 방향)
    for i in range(max_runs // 2):
        seeds.append(base_seed + (max_runs - i) * 2741)  # 다른 소수 간격
    
    # 3. 추가 랜덤 시드 (부족한 경우)
    np.random.seed(base_seed)
    while len(seeds) < max_runs:
        seeds.append(np.random.randint(1, 100000))
    
    # 시드 목록 자르기 및 셔플로 더 다양화
    seeds = seeds[:max_runs]
    np.random.seed(base_seed + 42)
    np.random.shuffle(seeds)
    
    print(f"Generated seeds for diversity: {seeds[:3]}{'...' if len(seeds) > 3 else ''}")
    
    # 적응적 조기 종료 조건
    early_stop_threshold = 0.02  # 2% 미만 개선이면 조기 종료 고려
    no_improvement_count = 0
    last_best_cost = float('inf')
    
    for run_idx in range(max_runs):
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # 동적 시간 제한 체크
        remaining_time = timelimit - elapsed_time
        remaining_runs = max_runs - run_idx
        
        # 남은 시간이 최소 실행 시간보다 적으면 종료
        if remaining_time < min_time_per_run:
            print(f"Time limit approaching ({remaining_time:.1f}s left). Stopping after {run_idx} runs.")
            break
        
        # 남은 실행들을 위한 시간이 부족하면 종료
        if remaining_runs > 1:
            avg_time_per_remaining = remaining_time / remaining_runs
            if avg_time_per_remaining < min_time_per_run * 0.8:  # 80% 여유도
                print(f"Insufficient time per remaining run ({avg_time_per_remaining:.1f}s). Stopping after {run_idx} runs.")
                break
        """
        # 적응적 조기 종료: 3회 이상 실행했고 개선이 미미하면 고려
        if run_idx >= 2 and len(all_costs) >= 2:
            if no_improvement_count >= 2 and remaining_time > timelimit * 0.3:  # 30% 이상 시간이 남았을 때만
                recent_improvement = (last_best_cost - best_cost) / last_best_cost if last_best_cost != float('inf') else 0
                if recent_improvement < early_stop_threshold:
                    print(f"Early stopping: minimal improvement ({recent_improvement*100:.1f}%) in recent runs.")
                    print(f"Saving remaining time for final optimization...")
                    break"""
        
        print(f"\n=== Run {run_idx + 1}/{max_runs} (Seed: {seeds[run_idx]}) ===")
        run_start_time = time.time()
        
        try:
            # 단일 알고리즘 실행
            current_solution = run_single_algorithm(prob_info, seeds[run_idx], run_idx)
            run_end_time = time.time()
            run_time = run_end_time - run_start_time
            
            # 솔루션 비용 계산
            feasibility_result = util.check_feasibility(prob_info, current_solution)
            
            if feasibility_result['feasible']:
                current_cost = feasibility_result['obj']
                all_costs.append(current_cost)
                
                print(f"Run {run_idx + 1} completed in {run_time:.2f}s - Cost: {current_cost}")
                
                # 최적 솔루션 업데이트
                if current_cost < best_cost:
                    improvement = ((best_cost - current_cost) / best_cost * 100) if best_cost != float('inf') else 0
                    last_best_cost = best_cost
                    best_cost = current_cost
                    best_solution = copy.deepcopy(current_solution)
                    best_run_info = {
                        'run': run_idx + 1,
                        'cost': current_cost,
                        'time': run_time,
                        'seed': seeds[run_idx],
                        'improvement': improvement
                    }
                    print(f"*** NEW BEST SOLUTION: Cost {current_cost} (-{improvement:.1f}%) ***")
                    no_improvement_count = 0  # 개선되었으므로 카운터 리셋
                else:
                    difference = ((current_cost - best_cost) / best_cost) * 100
                    print(f"Cost {current_cost} (+{difference:.1f}% vs best)")
                    no_improvement_count += 1
                    
                # 비용 분포 실시간 분석
                if len(all_costs) >= 2:
                    cost_std = np.std(all_costs)
                    cost_range = max(all_costs) - min(all_costs)
                    print(f"  Current statistics: Avg={np.mean(all_costs):.0f}, Std={cost_std:.0f}, Range={cost_range}")
                    
            else:
                print(f"Run {run_idx + 1} - INFEASIBLE SOLUTION")
                print(f"Infeasibility reasons: {feasibility_result.get('error', 'Unknown error')}")
                no_improvement_count += 1
                
        except Exception as e:
            print(f"Run {run_idx + 1} failed with exception: {str(e)}")
            no_improvement_count += 1
            continue
    
    # 결과 요약
    total_time = time.time() - start_time
    completed_runs = len(all_costs)
    
    print(f"\n=== OPTIMIZATION SUMMARY ===")
    print(f"Completed runs: {completed_runs}/{max_runs}")
    print(f"Total time: {total_time:.2f}s")
    
    if all_costs:
        print(f"Best cost: {best_cost} (Run {best_run_info['run']}, Seed: {best_run_info['seed']})")
        print(f"Worst cost: {max(all_costs)}")
        print(f"Average cost: {sum(all_costs)/len(all_costs):.2f}")
        print(f"Cost std dev: {np.std(all_costs):.2f}")
        print(f"Cost range: {max(all_costs) - min(all_costs)}")
        
        if len(all_costs) > 1:
            improvement_vs_first = ((all_costs[0] - best_cost) / all_costs[0]) * 100
            print(f"Improvement vs first run: {improvement_vs_first:.1f}%")
        
        print(f"All costs: {all_costs}")
        
        return best_solution
    else:
        print("No feasible solution found!")
        raise Exception("No feasible solution found in any run!")

if __name__ == "__main__":
    import json
    import os
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