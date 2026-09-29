# myalgorithm_enhanced_cluster_pathaware.py
# 기존 알고리즘 + 고도화된 경로 인식 클러스터 배치 최적화 + 미래 하역 고려 + 말단 노드 활용
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
    기존 알고리즘 + 고도화된 경로 인식 클러스터 배치 + 미래 하역 고려 + 말단 노드 활용
    """

    #------------- begin of improved algorithm code --------------#

    def identify_terminal_nodes(graph, node_degrees_threshold=2):
        """
        말단 노드 식별 (degree가 낮고 잎사귀에 가까운 노드들)
        """
        terminal_nodes = []
        node_degrees = dict(graph.degree())
        
        for node, degree in node_degrees.items():
            if degree <= node_degrees_threshold and node != 0:  # 게이트 제외
                terminal_nodes.append(node)
        
        # 거리별로 분류하여 반환
        terminal_by_distance = {}
        for node in terminal_nodes:
            try:
                distance = nx.shortest_path_length(graph, 0, node)
                if distance not in terminal_by_distance:
                    terminal_by_distance[distance] = []
                terminal_by_distance[distance].append(node)
            except:
                continue
                
        return terminal_nodes, terminal_by_distance

    def analyze_future_unloading_paths(prob_info, current_port):
        """
        현재 항구 이후의 모든 하역 수요에 대한 경로 분석
        미래에 내릴 화물들이 어떤 경로를 사용할지 예측
        """
        future_unloading_paths = {}
        path_usage_count = {}
        
        K = prob_info['K']
        P = prob_info['P']
        
        # 현재 항구 이후에 하역될 모든 수요 분석
        for k_idx, (origin_dest, quantity) in enumerate(K):
            origin, destination = origin_dest
            
            # 현재 항구에서 적재되고 미래에 하역될 화물
            if origin == current_port and destination > current_port:
                if destination not in future_unloading_paths:
                    future_unloading_paths[destination] = []
                
                # 이 수요에 대한 예상 하역 경로들을 저장
                future_unloading_paths[destination].extend([k_idx] * quantity)
        
        return future_unloading_paths

    def calculate_future_path_conflicts(graph, node_allocations, future_demands, gate_node=0):
        """
        미래 하역 시 경로 충돌 가능성 계산
        """
        conflict_scores = {}
        
        for dest, demand_list in future_demands.items():
            # 각 목적지별로 하역 시 사용될 가능성이 높은 경로들 분석
            occupied_nodes = [n for n, allocation in enumerate(node_allocations) if allocation != -1]
            
            # 현재 점유된 노드들 중에서 이 목적지 하역 시 경로상에 있을 가능성이 높은 노드들
            potential_blocking_nodes = []
            
            for node in occupied_nodes:
                try:
                    # 해당 노드에서 게이트까지의 경로
                    path_to_gate = nx.shortest_path(graph, node, gate_node)
                    
                    # 경로상의 다른 점유 노드들이 미래에 방해가 될 가능성
                    blocking_potential = 0
                    for path_node in path_to_gate[1:-1]:  # 시작점과 끝점 제외
                        if path_node in occupied_nodes:
                            blocking_potential += 1
                    
                    if blocking_potential > 0:
                        potential_blocking_nodes.append((node, blocking_potential))
                        
                except nx.NetworkXNoPath:
                    continue
            
            conflict_scores[dest] = potential_blocking_nodes
        
        return conflict_scores

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

    def find_path_aware_cluster_with_future_consideration(start_nodes, target_size, available_nodes, graph, 
                                                         critical_nodes, future_conflicts, terminal_nodes,
                                                         current_port, destination_port,
                                                         criticality_weight=10, connectivity_weight=-30, 
                                                         terminal_bonus=-5, future_penalty_weight=15, gate_node=0):
        """
        미래 하역을 고려한 경로 인식 클러스터 형성 + 말단 노드 활용 증대
        """
        if target_size <= 0:
            return []
        
        best_cluster = []
        best_score = float('inf')
        
        # 시작 노드들을 우선순위에 따라 정렬
        start_candidates = []
        for node in start_nodes:
            if node in available_nodes:
                criticality = critical_nodes.get(node, 0)
                is_terminal = node in terminal_nodes
                
                # 미래 충돌 가능성 평가
                future_conflict_score = 0
                if destination_port in future_conflicts:
                    for conflict_node, conflict_level in future_conflicts[destination_port]:
                        if node == conflict_node:
                            future_conflict_score = conflict_level
                
                try:
                    distance = nx.shortest_path_length(graph, gate_node, node)
                except:
                    distance = float('inf')
                
                # 말단 노드 보너스 적용
                terminal_adjustment = terminal_bonus if is_terminal else 0
                
                priority_score = (criticality * criticality_weight + 
                                future_conflict_score * future_penalty_weight +
                                terminal_adjustment)
                
                start_candidates.append((priority_score, distance, node, is_terminal))
        
        # 우선순위 순으로 정렬 (낮은 점수가 좋음)
        start_candidates.sort(key=lambda x: (x[0], x[1]))
        
        # 여러 시작점에서 클러스터 형성 시도
        for priority_score, distance, start_node, is_terminal in start_candidates[:min(5, len(start_candidates))]:
            cluster = build_cluster_with_future_awareness(start_node, target_size, available_nodes, 
                                                        graph, critical_nodes, terminal_nodes,
                                                        future_conflicts, destination_port,
                                                        criticality_weight, connectivity_weight,
                                                        terminal_bonus, future_penalty_weight, gate_node)
            
            if len(cluster) >= target_size:
                # 클러스터 품질 평가
                cluster_score = evaluate_cluster_quality_with_future(cluster, graph, critical_nodes, 
                                                                   terminal_nodes, available_nodes,
                                                                   future_conflicts, destination_port,
                                                                   criticality_weight, connectivity_weight,
                                                                   terminal_bonus, future_penalty_weight, gate_node)
                
                if cluster_score < best_score:
                    best_score = cluster_score
                    best_cluster = cluster
        
        return best_cluster[:target_size] if best_cluster else []

    def build_cluster_with_future_awareness(start_node, target_size, available_nodes, graph, critical_nodes, 
                                          terminal_nodes, future_conflicts, destination_port,
                                          criticality_weight, connectivity_weight, terminal_bonus, 
                                          future_penalty_weight, gate_node):
        """
        미래 하역 고려 및 말단 노드 우선 클러스터 구축
        """
        cluster = {start_node}
        queue = [start_node]
        
        # BFS로 클러스터 확장 (말단 노드 우선 고려)
        while queue and len(cluster) < target_size:
            current = queue.pop(0)
            
            # 이웃 노드들을 품질로 정렬
            neighbor_candidates = []
            for neighbor in graph.neighbors(current):
                if neighbor in available_nodes and neighbor not in cluster:
                    quality_score = calculate_node_quality_with_future(neighbor, cluster, available_nodes, 
                                                                     graph, critical_nodes, terminal_nodes,
                                                                     future_conflicts, destination_port,
                                                                     criticality_weight, connectivity_weight,
                                                                     terminal_bonus, future_penalty_weight, gate_node)
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

    def calculate_node_quality_with_future(node, current_cluster, available_nodes, graph, critical_nodes, 
                                         terminal_nodes, future_conflicts, destination_port,
                                         criticality_weight, connectivity_weight, terminal_bonus, 
                                         future_penalty_weight, gate_node):
        """
        미래 하역 고려 및 말단 노드 보너스를 적용한 노드 품질 계산
        """
        criticality = critical_nodes.get(node, 0)
        is_terminal = node in terminal_nodes
        
        # 미래 충돌 페널티
        future_conflict_penalty = 0
        if destination_port in future_conflicts:
            for conflict_node, conflict_level in future_conflicts[destination_port]:
                if node == conflict_node:
                    future_conflict_penalty = conflict_level * future_penalty_weight
        
        # 접근성 페널티 계산 (샘플링으로 성능 향상)
        accessibility_penalty = 0
        temp_unavailable = current_cluster | {node}
        
        test_nodes = [n for n in available_nodes if n not in temp_unavailable]
        if len(test_nodes) > 15:
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
                connectivity_bonus += 1
        
        # 게이트로부터의 거리 고려
        try:
            distance_to_gate = nx.shortest_path_length(graph, gate_node, node)
            optimal_distance = 6
            distance_penalty = abs(distance_to_gate - optimal_distance) * 0.5
        except:
            distance_penalty = 10
        
        # 말단 노드 보너스 적용
        terminal_adjustment = terminal_bonus if is_terminal else 0
        
        # 말단 노드인 경우 접근성 페널티 완화 (말단은 원래 막힐 가능성이 적음)
        if is_terminal:
            accessibility_penalty *= 0.5
        
        return (criticality * criticality_weight + 
                accessibility_penalty * 5 + 
                connectivity_bonus * connectivity_weight + 
                distance_penalty + 
                future_conflict_penalty +
                terminal_adjustment)

    def evaluate_cluster_quality_with_future(cluster, graph, critical_nodes, terminal_nodes, available_nodes,
                                           future_conflicts, destination_port, criticality_weight, 
                                           connectivity_weight, terminal_bonus, future_penalty_weight, gate_node):
        """
        미래 하역 고려 및 말단 노드 활용을 포함한 클러스터 품질 평가
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
        
        # 말단 노드 사용 보너스
        terminal_bonus_score = sum(terminal_bonus for node in cluster if node in terminal_nodes)
        
        # 미래 충돌 페널티
        future_conflict_penalty = 0
        if destination_port in future_conflicts:
            for node in cluster:
                for conflict_node, conflict_level in future_conflicts[destination_port]:
                    if node == conflict_node:
                        future_conflict_penalty += conflict_level * future_penalty_weight
        
        # 접근성 영향 페널티 (샘플링으로 효율화)
        temp_unavailable = set(cluster)
        remaining_nodes = [n for n in available_nodes if n not in temp_unavailable]
        blocked_count = 0
        
        sample_size = min(15, len(remaining_nodes))
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
        
        # 클러스터 응집도
        cluster_span = 0
        if len(cluster) > 1:
            distances = []
            for i, node1 in enumerate(cluster):
                for node2 in cluster[i+1:]:
                    try:
                        dist = nx.shortest_path_length(graph, node1, node2)
                        distances.append(dist)
                    except:
                        distances.append(10)
            cluster_span = sum(distances) / len(distances) if distances else 0
        
        return (critical_penalty * criticality_weight + 
                blocked_count * 20 + 
                connectivity_score * connectivity_weight + 
                cluster_span * 5 +
                future_conflict_penalty +
                terminal_bonus_score)

    def find_connected_cluster(start_nodes, target_size, available_nodes, graph):
        """
        시작 노드들로부터 BFS를 사용하여 연결된 클러스터를 찾는 함수 (fallback용)
        """
        if target_size <= 0:
            return []
        
        cluster = set()
        queue = list(start_nodes)
        
        for node in start_nodes:
            if node in available_nodes:
                cluster.add(node)
                if len(cluster) >= target_size:
                    break
        
        while queue and len(cluster) < target_size:
            current = queue.pop(0)
            
            for neighbor in graph.neighbors(current):
                if (neighbor in available_nodes and 
                    neighbor not in cluster and 
                    len(cluster) < target_size):
                    cluster.add(neighbor)
                    queue.append(neighbor)
        
        return list(cluster)

    def optimize_terminal_node_allocation(available_nodes, terminal_nodes, terminal_by_distance, 
                                        target_size, min_terminal_ratio=0.3):
        """
        말단 노드 할당 최적화 - 전체 할당에서 일정 비율의 말단 노드 사용 보장
        """
        target_terminal_count = min(len(terminal_nodes), int(target_size * min_terminal_ratio))
        
        if target_terminal_count == 0:
            return []
        
        # 거리별로 말단 노드 선택 (고르게 분산)
        selected_terminals = []
        distances = sorted(terminal_by_distance.keys())
        
        nodes_per_distance = max(1, target_terminal_count // len(distances)) if distances else 1
        remaining_count = target_terminal_count
        
        for distance in distances:
            if remaining_count <= 0:
                break
                
            available_terminals_at_distance = [n for n in terminal_by_distance[distance] 
                                             if n in available_nodes]
            
            take_count = min(nodes_per_distance, len(available_terminals_at_distance), remaining_count)
            selected_terminals.extend(available_terminals_at_distance[:take_count])
            remaining_count -= take_count
        
        # 부족한 경우 추가 선택
        if remaining_count > 0:
            for distance in distances:
                if remaining_count <= 0:
                    break
                    
                additional_terminals = [n for n in terminal_by_distance[distance] 
                                     if n in available_nodes and n not in selected_terminals]
                
                take_count = min(len(additional_terminals), remaining_count)
                selected_terminals.extend(additional_terminals[:take_count])
                remaining_count -= take_count
        
        return selected_terminals

    def loading_heuristic(p, node_allocations, rehandling_demands, node_priorities):
        print(f'Starting loading phase for port {p}...')

        # All demands that should be loaded at port p
        K_load = {idx: K[idx][1] for idx in range(len(K)) if K[idx][0][0] == p}
        print(f'Demands that are originated from this port: {K_load}')

        if len(rehandling_demands) > 0:
            print(f'Rehandled demands from unloading phase: {rehandling_demands}')
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
            print(f"Not enough available nodes ({len(available_nodes)} are available) to load {total_loading_demands} at port {p}.")
            raise Exception("Not enough available nodes to load demand!!! This must not happen!!!")

        if len(reachable_nodes) < total_loading_demands:
            print(f"Not enough reachable nodes ({len(reachable_nodes)} are reachable) to load {total_loading_demands} demands at port {p}. Rehandling...")

            # 기존 rehandling 로직
            available_but_not_reachable = [n for n in available_nodes if n not in reachable_nodes]
            
            priority_order = node_priorities['composite']
            available_but_not_reachable.sort(key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))

            while len(reachable_nodes) < total_loading_demands:
                if len(available_but_not_reachable) == 0:
                    print("Not enough available_but_not_reachable nodes to rehandle. This must not happen!!!")
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

            print(f'After rehandling, we have {len(reachable_nodes)} reachable nodes now! (but have to rehandle {len(last_rehandling_demands)} demands)')

        # Merge the rehandling demands with the loading demands
        for k in last_rehandling_demands:
            k = int(k)
            if k in K_load:
                K_load[k] += 1
            else:
                K_load[k] = 1

        print(f'total_loading_demands = {total_loading_demands}')

        if total_loading_demands > 0:
            # IMPROVED: 미래 하역 고려 + 말단 노드 적극 활용
            print("=== Applying future-aware path optimization with enhanced terminal node utilization ===")
            
            # 1. 말단 노드 식별
            terminal_nodes, terminal_by_distance = identify_terminal_nodes(G)
            print(f"Terminal nodes identified: {len(terminal_nodes)} nodes across {len(terminal_by_distance)} distance layers")
            
            # 2. 미래 하역 경로 분석
            future_unloading_demands = analyze_future_unloading_paths(prob_info, p)
            future_conflicts = calculate_future_path_conflicts(G, node_allocations, future_unloading_demands)
            
            print(f"Future unloading analysis: {len(future_unloading_demands)} destination ports to consider")
            if future_conflicts:
                total_conflicts = sum(len(conflicts) for conflicts in future_conflicts.values())
                print(f"Potential future conflicts: {total_conflicts} nodes may cause rehandling")
            
            # 3. 수요를 목적지별로 분류하고 역순 정렬
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
            print(f"Destination priority (latest first): {sorted_destinations}")
            
            # 4. 다음 항구(p+1) 수요 분리 및 우선 처리 계획
            next_port = p + 1
            next_port_demands = demands_by_destination.get(next_port, [])
            
            if next_port_demands:
                print(f"*** Next port priority: {len(next_port_demands)} containers for port {next_port} will be placed near gate ***")
                priority_destinations = [dest for dest in sorted_destinations if dest != next_port]
                priority_destinations.append(next_port)
            else:
                priority_destinations = sorted_destinations
            
            # 5. 중요 경로 노드들 식별
            critical_nodes = get_critical_path_nodes(G, reachable_nodes, 0)
            print(f"Critical path nodes identified: {len(critical_nodes)} nodes")
            
            # 6. 거리별 노드 분류
            reachable_nodes_by_distance = {}
            for n in reachable_nodes:
                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                if distance not in reachable_nodes_by_distance:
                    reachable_nodes_by_distance[distance] = []
                reachable_nodes_by_distance[distance].append(n)
            
            # 각 거리층을 중요도 고려하여 정렬 (덜 중요한 노드 우선, 말단 노드 우선)
            for distance in reachable_nodes_by_distance:
                nodes_with_priority = []
                for n in reachable_nodes_by_distance[distance]:
                    criticality = critical_nodes.get(n, 0)
                    is_terminal = n in terminal_nodes
                    # 말단 노드는 우선순위 boost
                    priority_score = criticality - (5 if is_terminal else 0)
                    nodes_with_priority.append((priority_score, n))
                nodes_with_priority.sort(key=lambda x: (x[0], x[1]))
                reachable_nodes_by_distance[distance] = [n for _, n in nodes_with_priority]
            
            print(f"Available nodes by distance (terminal nodes prioritized in each layer)")
            
            # 7. 말단 노드 활용 목표 설정
            terminal_utilization_target = min(len(terminal_nodes), int(total_loading_demands * 0.4))  # 40% 목표
            print(f"Terminal node utilization target: {terminal_utilization_target}/{total_loading_demands} ({terminal_utilization_target/total_loading_demands*100:.1f}%)")
            
            # 8. 미래 하역 고려 클러스터 할당
            loading_assignments = []
            used_nodes = set()
            terminal_nodes_used = 0
            
            for dest_idx, dest in enumerate(priority_destinations):
                dest_demands = demands_by_destination[dest]
                cluster_size = len(dest_demands)
                
                # 다음 항구 수요인지 확인
                is_next_port = (dest == next_port)
                
                if is_next_port:
                    print(f"  *** Next Port {dest}: {cluster_size} containers - near-gate placement ***")
                else:
                    print(f"  Destination {dest}: {cluster_size} containers - future-aware cluster with terminal preference")
                
                # 동적 가중치 조정
                remaining_demands = sum(len(demands_by_destination[d]) for d in priority_destinations[dest_idx:])
                remaining_space = len([n for n in reachable_nodes if n not in used_nodes])
                
                criticality_weight = 15 if remaining_space < remaining_demands * 1.2 else 10
                connectivity_weight = -40 if remaining_space < remaining_demands * 1.5 else -30
                terminal_bonus = -8  # 말단 노드 보너스 증가
                future_penalty_weight = 12
                
                # 말단 노드 사용률이 낮으면 보너스 증가
                current_terminal_ratio = terminal_nodes_used / max(1, len(loading_assignments))
                if current_terminal_ratio < 0.3:  # 30% 미만이면 말단 노드 더 선호
                    terminal_bonus = -15
                
                print(f"    Dynamic weights: criticality={criticality_weight}, terminal_bonus={terminal_bonus}")
                
                # 클러스터 형성
                cluster_nodes = []
                
                if is_next_port:
                    # 다음 항구: 가까운 곳부터 배치 (게이트 근처 우선)
                    distance_order = sorted(reachable_nodes_by_distance.keys())
                    print(f"    Next port strategy: placing near gate")
                    
                    for distance in distance_order:
                        available_start_nodes = [n for n in reachable_nodes_by_distance[distance] 
                                               if n not in used_nodes]
                        
                        if available_start_nodes:
                            remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                            cluster_nodes = find_connected_cluster(
                                available_start_nodes[:1],
                                cluster_size, 
                                remaining_available, 
                                G
                            )
                            if len(cluster_nodes) >= cluster_size:
                                break
                else:
                    # 일반 수요: 미래 하역 고려 + 말단 노드 우선 클러스터 형성
                    distance_order = sorted(reachable_nodes_by_distance.keys(), reverse=True)
                    
                    for distance in distance_order:
                        available_start_nodes = [n for n in reachable_nodes_by_distance[distance] 
                                               if n not in used_nodes]
                        
                        if available_start_nodes:
                            remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                            
                            cluster_nodes = find_path_aware_cluster_with_future_consideration(
                                available_start_nodes[:3],
                                cluster_size, 
                                remaining_available, 
                                G, 
                                critical_nodes,
                                future_conflicts,
                                terminal_nodes,
                                p,  # current_port
                                dest,  # destination_port
                                criticality_weight,
                                connectivity_weight,
                                terminal_bonus,
                                future_penalty_weight
                            )
                            if len(cluster_nodes) >= cluster_size:
                                break
                
                # 클러스터를 찾지 못한 경우, fallback 방식 사용
                if len(cluster_nodes) < cluster_size:
                    print(f"    Primary cluster formation failed, trying fallback methods...")
                    
                    if is_next_port:
                        # 다음 항구: 가까운 거리부터 순차적으로 할당
                        remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                        remaining_with_distance = []
                        for n in remaining_available:
                            distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                            is_terminal = n in terminal_nodes
                            # 말단 노드도 고려하되, 거리를 우선시
                            priority = (distance, -1 if is_terminal else 0, n)
                            remaining_with_distance.append(priority)
                        remaining_with_distance.sort(key=lambda x: (x[0], x[1]))
                        
                        cluster_nodes = [n for _, _, n in remaining_with_distance[:cluster_size]]
                        print(f"    Next port fallback: Distance-based allocation with terminal preference")
                    else:
                        # 일반 수요: 말단 노드 우선 fallback
                        remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                        
                        # 말단 노드를 우선적으로 할당
                        terminal_candidates = [n for n in remaining_available if n in terminal_nodes]
                        non_terminal_candidates = [n for n in remaining_available if n not in terminal_nodes]
                        
                        # 말단 노드를 거리순으로 정렬
                        terminal_with_distance = []
                        for n in terminal_candidates:
                            distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                            criticality = critical_nodes.get(n, 0)
                            terminal_with_distance.append((criticality, distance, n))
                        terminal_with_distance.sort(key=lambda x: (x[0], -x[1]))  # 중요도 낮은 것, 먼 것부터
                        
                        # 비말단 노드도 중요도순으로 정렬
                        non_terminal_with_criticality = []
                        for n in non_terminal_candidates:
                            criticality = critical_nodes.get(n, 0)
                            non_terminal_with_criticality.append((criticality, n))
                        non_terminal_with_criticality.sort(key=lambda x: x[0])
                        
                        # 말단 노드 우선 할당, 부족하면 비말단 노드 사용
                        cluster_nodes = []
                        terminal_allocation = min(cluster_size, len(terminal_with_distance), 
                                                max(1, int(cluster_size * 0.6)))  # 최대 60% 말단 노드
                        
                        for i in range(terminal_allocation):
                            if i < len(terminal_with_distance):
                                _, _, node = terminal_with_distance[i]
                                cluster_nodes.append(node)
                        
                        remaining_needed = cluster_size - len(cluster_nodes)
                        for i in range(remaining_needed):
                            if i < len(non_terminal_with_criticality):
                                _, node = non_terminal_with_criticality[i]
                                cluster_nodes.append(node)
                        
                        print(f"    Fallback: Terminal-priority allocation ({len([n for n in cluster_nodes if n in terminal_nodes])}/{len(cluster_nodes)} terminal)")
                
                # 최종 fallback: 순차 할당
                if len(cluster_nodes) < cluster_size:
                    remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                    additional_needed = cluster_size - len(cluster_nodes)
                    
                    additional_nodes = []
                    for n in remaining_available:
                        if n not in cluster_nodes and len(additional_nodes) < additional_needed:
                            additional_nodes.append(n)
                    
                    cluster_nodes.extend(additional_nodes)
                    print(f"    Final fallback: Sequential allocation for remaining {len(additional_nodes)} nodes")
                
                # 클러스터 내 정렬 및 할당
                cluster_nodes_with_info = []
                for n in cluster_nodes:
                    distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                    criticality = critical_nodes.get(n, 0)
                    is_terminal = n in terminal_nodes
                    cluster_nodes_with_info.append((distance, criticality, is_terminal, n))
                
                if is_next_port:
                    # 다음 항구: 가까운 순서 우선
                    cluster_nodes_with_info.sort(key=lambda x: (x[0], x[1], not x[2]))
                else:
                    # 일반 수요: 거리, 중요도, 말단 여부 고려
                    cluster_nodes_with_info.sort(key=lambda x: (x[0], x[1], not x[2]))
                
                # 할당 실행
                assigned_count = 0
                for i, (distance, criticality, is_terminal, node) in enumerate(cluster_nodes_with_info):
                    if assigned_count < len(dest_demands) and node not in used_nodes:
                        demand_k = dest_demands[assigned_count]
                        loading_assignments.append((node, demand_k))
                        used_nodes.add(node)
                        
                        if is_terminal:
                            terminal_nodes_used += 1
                        
                        if assigned_count < 3:  # 처음 3개만 로그
                            terminal_info = "(TERMINAL)" if is_terminal else ""
                            critical_info = f"(critical: {criticality})" if criticality > 0 else ""
                            print(f"    Demand {demand_k} -> Node {node} (dist {distance}) {terminal_info} {critical_info}")
                        
                        assigned_count += 1
                
                # 할당 부족 시 추가 처리
                if assigned_count < len(dest_demands):
                    remaining_demands = len(dest_demands) - assigned_count
                    print(f"    Need {remaining_demands} more nodes for destination {dest}")
                    
                    additional_nodes = []
                    for n in reachable_nodes:
                        if n not in used_nodes and len(additional_nodes) < remaining_demands:
                            distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                            criticality = critical_nodes.get(n, 0)
                            is_terminal = n in terminal_nodes
                            additional_nodes.append((distance, criticality, is_terminal, n))
                    
                    additional_nodes.sort(key=lambda x: (x[0], x[1], not x[2]))
                    
                    for distance, criticality, is_terminal, node in additional_nodes:
                        if assigned_count < len(dest_demands):
                            demand_k = dest_demands[assigned_count]
                            loading_assignments.append((node, demand_k))
                            used_nodes.add(node)
                            
                            if is_terminal:
                                terminal_nodes_used += 1
                            
                            terminal_info = "(TERMINAL)" if is_terminal else ""
                            critical_info = f"(critical: {criticality})" if criticality > 0 else ""
                            print(f"    Additional: Demand {demand_k} -> Node {node} (dist {distance}) {terminal_info} {critical_info}")
                            assigned_count += 1
                
                # 클러스터 품질 리포트
                cluster_terminal_count = sum(1 for _, _, is_terminal, _ in cluster_nodes_with_info if is_terminal)
                cluster_critical_count = sum(1 for _, c, _, _ in cluster_nodes_with_info if c > 0)
                
                if is_next_port:
                    avg_distance = sum(d for d, _, _, _ in cluster_nodes_with_info) / len(cluster_nodes_with_info) if cluster_nodes_with_info else 0
                    print(f"    *** Next port cluster: {len(cluster_nodes)} nodes, avg distance: {avg_distance:.1f}, {cluster_terminal_count} terminal, {cluster_critical_count} critical ***")
                else:
                    print(f"    Cluster formed: {len(cluster_nodes)} nodes, {cluster_terminal_count} terminal, {cluster_critical_count} critical")

            # 최종 중복 검증
            node_usage = {}
            duplicate_found = False
            for node, demand_k in loading_assignments:
                if node in node_usage:
                    print(f"ERROR: Node {node} is assigned to multiple demands: {node_usage[node]} and {demand_k}")
                    duplicate_found = True
                else:
                    node_usage[node] = demand_k
            
            if duplicate_found:
                print("CRITICAL: Duplicate node assignments detected. Fixing...")
                unique_assignments = []
                used_in_final = set()
                missing_demands = []
                
                for node, demand_k in loading_assignments:
                    if node not in used_in_final:
                        unique_assignments.append((node, demand_k))
                        used_in_final.add(node)
                    else:
                        missing_demands.append(demand_k)
                
                for demand_k in missing_demands:
                    for n in reachable_nodes:
                        if n not in used_in_final:
                            unique_assignments.append((n, demand_k))
                            used_in_final.add(n)
                            print(f"    Fixed: Demand {demand_k} reassigned to Node {n}")
                            break
                
                loading_assignments = unique_assignments
                used_nodes = used_in_final

            assert(len(loading_assignments) == total_loading_demands)

            # 9. 할당 후 접근성 및 미래 충돌 검증
            remaining_nodes = [n for n in reachable_nodes if n not in used_nodes]
            blocked_nodes = []
            for node in remaining_nodes:
                temp_graph = G.copy()
                temp_graph.remove_nodes_from(used_nodes)
                if not nx.has_path(temp_graph, 0, node):
                    blocked_nodes.append(node)
            
            # 미래 충돌 예상 분석
            future_rehandling_risk = 0
            for dest, conflicts in future_conflicts.items():
                for conflict_node, conflict_level in conflicts:
                    if conflict_node in used_nodes:
                        future_rehandling_risk += conflict_level
            
            if blocked_nodes:
                print(f"    Warning: {len(blocked_nodes)} nodes became unreachable after allocation")
                
                # 적응적 조정 (기존과 동일하지만 말단 노드 고려)
                if len(blocked_nodes) > len(reachable_nodes) * 0.1:
                    print(f"    Attempting adaptive adjustment with terminal node preference...")
                    
                    adjustable_assignments = []
                    for idx, (node, demand_k) in enumerate(loading_assignments):
                        node_criticality = critical_nodes.get(node, 0)
                        is_terminal = node in terminal_nodes
                        
                        if node_criticality > 0 and not is_terminal:  # 중요하면서 말단이 아닌 노드만 조정 대상
                            alternative_nodes = [n for n in remaining_nodes + blocked_nodes 
                                               if (critical_nodes.get(n, 0) < node_criticality or n in terminal_nodes)]
                            if alternative_nodes:
                                adjustable_assignments.append((idx, node, demand_k, node_criticality, alternative_nodes))
                    
                    adjustable_assignments.sort(key=lambda x: x[3], reverse=True)
                    
                    adjustments_made = 0
                    already_used_alternatives = set()
                    
                    for idx, old_node, demand_k, criticality, alternatives in adjustable_assignments[:3]:
                        best_alternative = None
                        best_score = float('inf')
                        
                        available_alternatives = [alt for alt in alternatives[:10] 
                                                if alt not in already_used_alternatives and alt not in used_nodes]
                        
                        for alt_node in available_alternatives:
                            temp_used = (used_nodes - {old_node}) | {alt_node}
                            temp_remaining = [n for n in reachable_nodes if n not in temp_used]
                            temp_blocked = 0
                            
                            for test_node in temp_remaining[:10]:
                                temp_graph = G.copy()
                                temp_graph.remove_nodes_from(temp_used)
                                try:
                                    if not nx.has_path(temp_graph, 0, test_node):
                                        temp_blocked += 1
                                except:
                                    temp_blocked += 1
                            
                            # 말단 노드는 보너스
                            terminal_bonus_score = -2 if alt_node in terminal_nodes else 0
                            score = temp_blocked * 10 + critical_nodes.get(alt_node, 0) + terminal_bonus_score
                            
                            if score < best_score:
                                best_score = score
                                best_alternative = alt_node
                        
                        if best_alternative and best_alternative not in used_nodes:
                            loading_assignments[idx] = (best_alternative, demand_k)
                            used_nodes.remove(old_node)
                            used_nodes.add(best_alternative)
                            already_used_alternatives.add(best_alternative)
                            adjustments_made += 1
                            
                            is_terminal_alt = best_alternative in terminal_nodes
                            is_terminal_old = old_node in terminal_nodes
                            
                            if is_terminal_alt and not is_terminal_old:
                                terminal_nodes_used += 1
                            elif not is_terminal_alt and is_terminal_old:
                                terminal_nodes_used -= 1
                            
                            print(f"      Adjusted: Node {old_node} -> Node {best_alternative} {'(TERMINAL)' if is_terminal_alt else ''}")
                            
                            if adjustments_made >= 3:
                                break
                    
                    if adjustments_made > 0:
                        final_remaining = [n for n in reachable_nodes if n not in used_nodes]
                        final_blocked = []
                        for node in final_remaining:
                            temp_graph = G.copy()
                            temp_graph.remove_nodes_from(used_nodes)
                            if not nx.has_path(temp_graph, 0, node):
                                final_blocked.append(node)
                        
                        print(f"    After adjustment: {len(final_blocked)} nodes blocked (improved by {len(blocked_nodes) - len(final_blocked)})")
            else:
                print(f"    Excellent: No nodes blocked after allocation")

            # 10. 최종 할당 품질 리포트
            total_critical_used = sum(1 for node, _ in loading_assignments if critical_nodes.get(node, 0) > 0)
            total_terminal_used = sum(1 for node, _ in loading_assignments if node in terminal_nodes)
            avg_distance = sum(len(shortest_paths[node][0]) - 1 for node, _ in loading_assignments) / len(loading_assignments)
            
            # 클러스터 응집도 계산
            cluster_quality_scores = []
            for dest in sorted_destinations:
                dest_nodes = [node for node, demand_k in loading_assignments if int(demand_k) in demands_by_destination[dest]]
                if len(dest_nodes) > 1:
                    internal_edges = sum(1 for i, n1 in enumerate(dest_nodes) for n2 in dest_nodes[i+1:] if G.has_edge(n1, n2))
                    max_edges = len(dest_nodes) * (len(dest_nodes) - 1) / 2
                    cohesion = internal_edges / max_edges if max_edges > 0 else 0
                    cluster_quality_scores.append(cohesion)
            
            avg_cohesion = sum(cluster_quality_scores) / len(cluster_quality_scores) if cluster_quality_scores else 0
            
            print(f"    Final allocation quality:")
            print(f"      Critical nodes used: {total_critical_used}/{len(loading_assignments)} ({total_critical_used/len(loading_assignments)*100:.1f}%)")
            print(f"      Terminal nodes used: {total_terminal_used}/{len(loading_assignments)} ({total_terminal_used/len(loading_assignments)*100:.1f}%)")
            print(f"      Terminal utilization vs target: {total_terminal_used}/{terminal_utilization_target}")
            print(f"      Average distance: {avg_distance:.1f}")
            print(f"      Average cluster cohesion: {avg_cohesion:.2f}")
            print(f"      Future rehandling risk score: {future_rehandling_risk}")
            print(f"      Nodes blocked: {len(blocked_nodes) if 'blocked_nodes' in locals() else 0}")

            # 11. 실제 할당 및 경로 생성
            distances, previous_nodes = util.dijkstra(G, node_allocations)

            for n, k in loading_assignments:
                node_allocations[n] = k
                path = util.path_backtracking(previous_nodes, 0, n)
                route_list.append((path, k))

        print(f'Loading phase completed with {len(route_list)} routes including {len(last_rehandling_demands)} rehandling routes!')
        return route_list, node_allocations

    def unloading_heuristic(p, node_allocations, node_priorities):
        # All demands that should be unloaded at port p
        K_unload = {idx: K[idx][1] for idx in range(len(K)) if K[idx][0][1] == p}

        print(f'Starting unloading phase...')
        print(f'Demands that are destined to this port: {K_unload}')

        route_list = []
        rehandling_demands = []

        # 거리 순으로 unloading 처리 (가까운 것부터) - 말단 노드 우선 고려
        all_unloading_targets = []
        terminal_nodes, _ = identify_terminal_nodes(G)
        
        for k, r in K_unload.items():
            unloading_nodes = [n for n in range(N) if node_allocations[n] == k]
            for n in unloading_nodes:
                if node_allocations[n] != -1:
                    distance_to_gate = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                    is_terminal = n in terminal_nodes
                    # 말단 노드는 우선순위 boost (더 쉽게 접근 가능)
                    priority_score = distance_to_gate - (0.5 if is_terminal else 0)
                    all_unloading_targets.append((priority_score, n, k, is_terminal))
        
        all_unloading_targets.sort(key=lambda x: x[0])  # 우선순위 높은 것부터
        
        terminal_unloaded = 0
        for priority_score, n, k, is_terminal in all_unloading_targets:
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
                if is_terminal:
                    terminal_unloaded += 1
            else:
                blocking_nodes = min_blocking_nodes[2]
                path = min_blocking_nodes[3]
                
                blocking_nodes_with_distance = []
                for bn in blocking_nodes:
                    if node_allocations[bn] != -1:
                        bn_distance = len(shortest_paths[bn][0]) - 1 if shortest_paths[bn] else float('inf')
                        bn_is_terminal = bn in terminal_nodes
                        # 말단 노드 블로커는 더 높은 우선순위로 먼저 제거
                        bn_priority = bn_distance - (0.3 if bn_is_terminal else 0)
                        blocking_nodes_with_distance.append((bn_priority, bn, bn_is_terminal))
                
                blocking_nodes_with_distance.sort(key=lambda x: x[0])
                
                for bn_priority, bn, bn_is_terminal in blocking_nodes_with_distance:
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
                if is_terminal:
                    terminal_unloaded += 1

        print(f'Unloading phase completed: {len(route_list)} routes, {len(rehandling_demands)} rehandling, {terminal_unloaded} terminal nodes unloaded')
        return route_list, rehandling_demands

    # Get parameters from the prob_info dictionary
    N = prob_info['N']
    E = prob_info['E']
    E = set([(u,v) for (u,v) in E])
    K = prob_info['K']
    P = prob_info['P']
    F = prob_info['F']

    start_time = time.time()

    # Create a graph for the problem
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(E)

    # 노드 우선순위 초기화
    print("Initializing node priorities...")
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
        print(f"Port {p} ==============================")

        if p > 0:
            # Unloading heuristic
            route_list_unload, rehandling_demands = unloading_heuristic(p, node_allocations, node_priorities)
            solution[p].extend(route_list_unload)
        else:
            rehandling_demands = []

        if p < P-1:
            # Loading heuristic with enhanced path-aware cluster placement and future consideration
            route_list_load, node_allocations = loading_heuristic(p, node_allocations, rehandling_demands, node_priorities)
            solution[p].extend(route_list_load)

        print()

    # Path 정렬 - 충돌 최소화를 위한 고도화된 정렬
    print("Sorting paths to minimize conflicts...")
    for p in range(P):
        if solution[p]:
            loading_routes = []
            unloading_routes = []
            rehandling_routes = []
            
            for route, demand in solution[p]:
                if len(route) >= 2:
                    if route[0] == 0:  # Loading route
                        distance = len(route) - 1
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
            
            print(f"Port {p} route summary: {len(unloading_routes)} unloading, {len(rehandling_routes)} rehandling, {len(loading_routes)} loading")

    #------------- end of improved algorithm code --------------#

    return solution

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