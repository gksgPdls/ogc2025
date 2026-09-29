# myalgorithm_enhanced_cluster_pathaware.py
# 기존 알고리즘 + 고도화된 경로 인식 클러스터 배치 최적화
# Last updated: 2025/07/25

import time
import util
import networkx as nx
from itertools import islice
import numpy as np
import priority
import random
from collections import defaultdict
import math

def algorithm(prob_info, timelimit=60):
    """
    기존 알고리즘 + 고도화된 경로 인식 클러스터 배치 (길막힘 방지 + 동적 최적화)
    """
    # 개선된 다중 클러스터 형성 전략

    class MultiClusterStrategy:
        """수요량과 네트워크 구조에 따른 다중 클러스터 형성 전략"""
        
        def __init__(self, network_analyzer, adaptive_weights):
            self.network_analyzer = network_analyzer
            self.adaptive_weights = adaptive_weights
            self.cluster_size_thresholds = self._calculate_optimal_thresholds()
        
        def _calculate_optimal_thresholds(self):
            """네트워크 타입별 최적 클러스터 크기 임계값 계산"""
            network_type = self.network_analyzer.network_metrics['network_type']
            network_size = len(self.network_analyzer.graph.nodes())
            
            # 네트워크 타입별 기본 설정
            base_thresholds = {
                'tree': {
                    'min_cluster_size': 2,
                    'max_cluster_size': max(4, network_size // 15),
                    'split_threshold': 6,  # 6개 이상이면 분할 고려
                    'density_factor': 0.15  # 네트워크 대비 15% 이상이면 분할
                },
                'linear': {
                    'min_cluster_size': 3,
                    'max_cluster_size': max(5, network_size // 12),
                    'split_threshold': 8,
                    'density_factor': 0.12
                },
                'grid': {
                    'min_cluster_size': 2,
                    'max_cluster_size': max(6, network_size // 10),
                    'split_threshold': 10,
                    'density_factor': 0.10
                },
                'complex': {
                    'min_cluster_size': 3,
                    'max_cluster_size': max(8, network_size // 8),
                    'split_threshold': 12,
                    'density_factor': 0.08
                }
            }
            
            return base_thresholds.get(network_type, base_thresholds['grid'])
        
        def should_split_demand(self, demand_size, network_size, remaining_space, is_next_port=False):
            """수요를 여러 클러스터로 분할해야 하는지 판단"""
            if is_next_port:
                return False  # 다음 항구는 분할하지 않음 (빠른 접근 우선)
            
            thresholds = self.cluster_size_thresholds
            
            # 1. 절대적 크기 기준
            if demand_size >= thresholds['split_threshold']:
                return True
            
            # 2. 네트워크 대비 비율 기준
            density_ratio = demand_size / network_size
            if density_ratio >= thresholds['density_factor']:
                return True
            
            # 3. 남은 공간 대비 비율 기준
            if remaining_space > 0 and demand_size / remaining_space >= 0.3:
                return True
            
            return False
        
        def calculate_optimal_cluster_count(self, demand_size, network_size, remaining_space, is_next_port=False):
            """최적 클러스터 개수 계산"""
            if not self.should_split_demand(demand_size, network_size, remaining_space, is_next_port):
                return 1
            
            thresholds = self.cluster_size_thresholds
            max_cluster_size = thresholds['max_cluster_size']
            min_cluster_size = thresholds['min_cluster_size']
            
            # 기본적으로는 max_cluster_size로 나누어 계산
            basic_count = max(1, (demand_size + max_cluster_size - 1) // max_cluster_size)
            
            # 네트워크 구조와 남은 공간을 고려한 조정
            space_pressure = demand_size / max(remaining_space, 1)
            
            if space_pressure > 1.5:  # 공간이 부족하면 더 많은 클러스터로 분산
                adjusted_count = min(basic_count + 1, demand_size // min_cluster_size)
            elif space_pressure < 0.5:  # 공간이 충분하면 클러스터 수 줄임
                adjusted_count = max(1, basic_count - 1)
            else:
                adjusted_count = basic_count
            
            # 최종 검증: 각 클러스터가 최소 크기를 만족하는지 확인
            if adjusted_count * min_cluster_size > demand_size:
                adjusted_count = max(1, demand_size // min_cluster_size)
            
            return min(adjusted_count, demand_size)  # 수요보다 많은 클러스터는 의미없음
        
        def distribute_demands_to_clusters(self, demand_list, cluster_count):
            """수요를 여러 클러스터에 균등하게 분배"""
            if cluster_count <= 1:
                return [demand_list]
            
            # 균등 분배
            base_size = len(demand_list) // cluster_count
            extra = len(demand_list) % cluster_count
            
            clusters = []
            start_idx = 0
            
            for i in range(cluster_count):
                # 나머지는 앞쪽 클러스터에 하나씩 추가
                cluster_size = base_size + (1 if i < extra else 0)
                end_idx = start_idx + cluster_size
                
                if start_idx < len(demand_list):
                    clusters.append(demand_list[start_idx:end_idx])
                
                start_idx = end_idx
            
            return [cluster for cluster in clusters if cluster]  # 빈 클러스터 제거

    def enhanced_multi_cluster_loading_heuristic(p, node_allocations, rehandling_demands, node_priorities):
        """개선된 다중 클러스터 기반 적재 휴리스틱"""
        print(f'Starting enhanced multi-cluster loading phase...')

        # 기존 수요 처리 로직
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

        # 기존 초기화 코드...
        reachable_nodes, reachable_node_distances = util.bfs(G, node_allocations)
        available_nodes = util.get_available_nodes(node_allocations)

        if len(available_nodes) < total_loading_demands:
            print(f"Not enough available nodes ({len(available_nodes)} are available) to load {total_loading_demands} at port {p}.")
            raise Exception("Not enough available nodes to load demand!!! This must not happen!!!")

        # 기존 rehandling 로직... (생략)

        if total_loading_demands > 0:
            print("=== Applying enhanced multi-cluster allocation strategy ===")
            
            # 네트워크 분석 시스템 초기화
            network_analyzer = NetworkAnalyzer(G, 0)
            adaptive_weights = AdaptiveWeightSystem(network_analyzer)
            multi_cluster_strategy = MultiClusterStrategy(network_analyzer, adaptive_weights)
            
            print(f"Network analysis: type={network_analyzer.network_metrics['network_type']}, "
                f"size={len(G.nodes())}, "
                f"terminal_regions={len(network_analyzer.network_metrics['terminal_regions'])}")
            
            # 1. 수요를 목적지별로 분류하고 역순 정렬
            demands_by_destination = {}
            for k, qty in K_load.items():
                k = int(k)
                destination = K[k][0][1] 
                if destination not in demands_by_destination:
                    demands_by_destination[destination] = []
                for _ in range(qty):
                    demands_by_destination[destination].append(k)
            
            sorted_destinations = sorted(demands_by_destination.keys(), reverse=True)
            print(f"Destination priority (latest first): {sorted_destinations}")
            
            # 다음 항구 수요 분리
            next_port = p + 1
            next_port_demands = demands_by_destination.get(next_port, [])
            
            if next_port_demands:
                print(f"*** Next port priority: {len(next_port_demands)} containers for port {next_port} ***")
                priority_destinations = [dest for dest in sorted_destinations if dest != next_port]
                priority_destinations.append(next_port)
            else:
                priority_destinations = sorted_destinations
            
            # 2. 중요 경로 노드들 식별
            critical_nodes = get_critical_path_nodes(G, reachable_nodes, 0)
            print(f"Critical path nodes identified: {len(critical_nodes)} nodes")
            
            # 3. 거리별 노드 분류
            reachable_nodes_by_distance = {}
            for n in reachable_nodes:
                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                if distance not in reachable_nodes_by_distance:
                    reachable_nodes_by_distance[distance] = []
                reachable_nodes_by_distance[distance].append(n)
            
            # 중요도 고려 정렬
            for distance in reachable_nodes_by_distance:
                nodes_with_criticality = []
                for n in reachable_nodes_by_distance[distance]:
                    criticality = critical_nodes.get(n, 0)
                    nodes_with_criticality.append((criticality, n))
                nodes_with_criticality.sort(key=lambda x: (x[0], x[1]))
                reachable_nodes_by_distance[distance] = [n for _, n in nodes_with_criticality]
            
            # 4. *** 새로운 다중 클러스터 할당 로직 ***
            loading_assignments = []
            used_nodes = set()
            total_clusters_formed = 0
            
            for dest_idx, dest in enumerate(priority_destinations):
                dest_demands = demands_by_destination[dest]
                total_demand_size = len(dest_demands)
                
                is_next_port = (dest == next_port)
                remaining_demands = sum(len(demands_by_destination[d]) for d in priority_destinations[dest_idx:])
                remaining_space = len([n for n in reachable_nodes if n not in used_nodes])
                
                print(f"\n  === Destination {dest}: {total_demand_size} containers ===")
                print(f"    Remaining space: {remaining_space}, remaining demands: {remaining_demands}")
                
                # 다중 클러스터 전략 적용
                cluster_count = multi_cluster_strategy.calculate_optimal_cluster_count(
                    total_demand_size, len(G.nodes()), remaining_space, is_next_port
                )
                
                print(f"    Optimal cluster count: {cluster_count}")
                
                if cluster_count > 1:
                    print(f"    *** Multi-cluster strategy: splitting {total_demand_size} demands into {cluster_count} clusters ***")
                
                # 수요를 클러스터별로 분배
                demand_clusters = multi_cluster_strategy.distribute_demands_to_clusters(dest_demands, cluster_count)
                
                print(f"    Cluster sizes: {[len(cluster) for cluster in demand_clusters]}")
                
                # 각 클러스터별로 할당 수행
                for cluster_idx, cluster_demands in enumerate(demand_clusters):
                    cluster_size = len(cluster_demands)
                    
                    print(f"    --- Cluster {cluster_idx + 1}/{cluster_count}: {cluster_size} containers ---")
                    
                    # 현재 상황 업데이트
                    current_remaining_space = len([n for n in reachable_nodes if n not in used_nodes])
                    current_remaining_demands = sum(len(demands_by_destination[d]) for d in priority_destinations[dest_idx:]) - sum(len(dc) for dc in demand_clusters[:cluster_idx])
                    
                    current_situation = {
                        'remaining_space': current_remaining_space,
                        'remaining_demands': current_remaining_demands,
                        'destination': dest,
                        'is_next_port': is_next_port,
                        'used_nodes': used_nodes,
                        'available_nodes': [n for n in reachable_nodes if n not in used_nodes],
                        'critical_nodes': critical_nodes,
                        'demand_density': cluster_size / max(current_remaining_space, 1),
                        'cluster_index': cluster_idx,
                        'total_clusters': cluster_count
                    }
                    
                    # 클러스터별 시작점 선택 전략
                    if is_next_port:
                        # 다음 항구: 모든 클러스터를 가까운 곳에
                        distance_order = sorted(reachable_nodes_by_distance.keys())
                    else:
                        # 일반 수요: 클러스터별로 다른 거리 영역 활용
                        if cluster_count > 1:
                            # 여러 클러스터인 경우: 거리별로 분산 배치
                            available_distances = sorted(reachable_nodes_by_distance.keys(), reverse=True)
                            if cluster_idx < len(available_distances):
                                # 각 클러스터를 다른 거리 영역에 우선 배치
                                preferred_distance = available_distances[cluster_idx % len(available_distances)]
                                distance_order = [preferred_distance] + [d for d in available_distances if d != preferred_distance]
                                print(f"      Cluster {cluster_idx + 1} targeting distance layer {preferred_distance}")
                            else:
                                distance_order = available_distances
                        else:
                            # 단일 클러스터: 기존 방식
                            distance_order = sorted(reachable_nodes_by_distance.keys(), reverse=True)
                    
                    # 클러스터 형성
                    cluster_nodes = []
                    
                    for distance in distance_order:
                        available_start_nodes = [n for n in reachable_nodes_by_distance[distance] 
                                            if n not in used_nodes]
                        
                        if available_start_nodes:
                            remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                            
                            if is_next_port:
                                # 다음 항구: 간단한 연결된 클러스터
                                cluster_nodes = find_connected_cluster(
                                    available_start_nodes[:1],
                                    cluster_size, 
                                    remaining_available, 
                                    G
                                )
                            else:
                                # 일반 수요: 고도화된 경로 인식 클러스터
                                cluster_nodes = find_path_aware_cluster_enhanced(
                                    available_start_nodes[:2],  # 다중 클러스터이므로 후보 수 줄임
                                    cluster_size, 
                                    remaining_available, 
                                    G, 
                                    critical_nodes,
                                    network_analyzer,
                                    adaptive_weights,
                                    current_situation
                                )
                            
                            if len(cluster_nodes) >= cluster_size:
                                print(f"      Cluster formed at distance {distance} with {len(cluster_nodes)} nodes")
                                break
                    
                    # Fallback 처리
                    if len(cluster_nodes) < cluster_size:
                        print(f"      Primary formation failed, using fallback...")
                        remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                        
                        if is_next_port:
                            # 거리 순 할당
                            remaining_with_distance = []
                            for n in remaining_available:
                                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                                remaining_with_distance.append((distance, n))
                            remaining_with_distance.sort(key=lambda x: x[0])
                            cluster_nodes = [n for _, n in remaining_with_distance[:cluster_size]]
                        else:
                            # 중요도 순 할당
                            remaining_with_criticality = []
                            for n in remaining_available:
                                criticality = critical_nodes.get(n, 0)
                                remaining_with_criticality.append((criticality, n))
                            remaining_with_criticality.sort(key=lambda x: x[0])
                            cluster_nodes = [n for _, n in remaining_with_criticality[:cluster_size]]
                    
                    # 클러스터 내 노드 정렬 및 할당
                    cluster_nodes_with_distance = []
                    for n in cluster_nodes:
                        distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                        criticality = critical_nodes.get(n, 0)
                        cluster_nodes_with_distance.append((distance, criticality, n))
                    
                    if is_next_port:
                        cluster_nodes_with_distance.sort(key=lambda x: (x[0], x[1]))  # 가까운 순서
                    else:
                        cluster_nodes_with_distance.sort(key=lambda x: (x[0], x[1]))  # 거리, 중요도 순
                    
                    # 실제 할당
                    assigned_count = 0
                    for distance, criticality, node in cluster_nodes_with_distance:
                        if assigned_count < len(cluster_demands) and node not in used_nodes:
                            demand_k = cluster_demands[assigned_count]
                            loading_assignments.append((node, demand_k))
                            used_nodes.add(node)
                            
                            if assigned_count < 2:  # 처음 2개만 로그
                                critical_info = f"(critical: {criticality})" if criticality > 0 else ""
                                print(f"      Demand {demand_k} -> Node {node} (dist {distance}) {critical_info}")
                            
                            assigned_count += 1
                    
                    # 부족한 경우 추가 할당
                    if assigned_count < len(cluster_demands):
                        remaining_demands_count = len(cluster_demands) - assigned_count
                        additional_nodes = []
                        for n in reachable_nodes:
                            if n not in used_nodes and len(additional_nodes) < remaining_demands_count:
                                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                                criticality = critical_nodes.get(n, 0)
                                additional_nodes.append((distance, criticality, n))
                        
                        additional_nodes.sort(key=lambda x: (x[0], x[1]))
                        
                        for distance, criticality, node in additional_nodes:
                            if assigned_count < len(cluster_demands):
                                demand_k = cluster_demands[assigned_count]
                                loading_assignments.append((node, demand_k))
                                used_nodes.add(node)
                                print(f"      Additional: Demand {demand_k} -> Node {node} (dist {distance})")
                                assigned_count += 1
                    
                    total_clusters_formed += 1
                    
                print(f"    Destination {dest} completed: {cluster_count} clusters formed")
            
            # 5. 최종 검증 및 품질 리포트
            print(f"\n=== Multi-cluster allocation completed ===")
            print(f"Total clusters formed: {total_clusters_formed}")
            print(f"Average cluster size: {len(loading_assignments) / max(total_clusters_formed, 1):.1f}")
            
            # 중복 검증
            node_usage = {}
            duplicate_found = False
            for node, demand_k in loading_assignments:
                if node in node_usage:
                    print(f"ERROR: Node {node} is assigned to multiple demands")
                    duplicate_found = True
                else:
                    node_usage[node] = demand_k
            
            if not duplicate_found:
                print("✓ No duplicate assignments found")
            
            # 접근성 검증
            remaining_nodes = [n for n in reachable_nodes if n not in used_nodes]
            blocked_nodes = []
            for node in remaining_nodes:
                temp_graph = G.copy()
                temp_graph.remove_nodes_from(used_nodes)
                if not nx.has_path(temp_graph, 0, node):
                    blocked_nodes.append(node)
            
            print(f"Accessibility impact: {len(blocked_nodes)} nodes blocked")
            
            # 클러스터 품질 분석
            cluster_quality_by_dest = {}
            for dest in priority_destinations:
                dest_nodes = [node for node, demand_k in loading_assignments 
                            if int(demand_k) in demands_by_destination[dest]]
                if len(dest_nodes) > 1:
                    internal_edges = sum(1 for i, n1 in enumerate(dest_nodes) 
                                    for n2 in dest_nodes[i+1:] if G.has_edge(n1, n2))
                    max_edges = len(dest_nodes) * (len(dest_nodes) - 1) / 2
                    cohesion = internal_edges / max_edges if max_edges > 0 else 0
                    cluster_quality_by_dest[dest] = cohesion
            
            avg_cohesion = sum(cluster_quality_by_dest.values()) / len(cluster_quality_by_dest) if cluster_quality_by_dest else 0
            print(f"Average cluster cohesion: {avg_cohesion:.3f}")
            
            # 거리 분산도 분석
            distances_by_dest = {}
            for dest in priority_destinations:
                dest_nodes = [node for node, demand_k in loading_assignments 
                            if int(demand_k) in demands_by_destination[dest]]
                distances = []
                for node in dest_nodes:
                    distance = len(shortest_paths[node][0]) - 1 if shortest_paths[node] else float('inf')
                    distances.append(distance)
                if distances:
                    distances_by_dest[dest] = {
                        'avg': sum(distances) / len(distances),
                        'std': np.std(distances) if len(distances) > 1 else 0
                    }
            
            print(f"Distance distribution by destination:")
            for dest, stats in distances_by_dest.items():
                print(f"  Dest {dest}: avg={stats['avg']:.1f}, std={stats['std']:.1f}")
            
            assert(len(loading_assignments) == total_loading_demands)
            
            # 6. 실제 할당 및 경로 생성
            distances, previous_nodes = util.dijkstra(G, node_allocations)
            
            for n, k in loading_assignments:
                node_allocations[n] = k
                path = util.path_backtracking(previous_nodes, 0, n)
                route_list.append((path, k))

        print(f'Enhanced multi-cluster loading phase completed with {len(route_list)} routes!')
        return route_list, node_allocations
    class NetworkAnalyzer:
        """네트워크 구조 분석 및 특성 파악"""
        
        def __init__(self, graph, gate_node=0):
            self.graph = graph
            self.gate_node = gate_node
            self.network_metrics = {}
            self._analyze_network_structure()
        
        def _analyze_network_structure(self):
            """네트워크 구조 분석"""
            G = self.graph
            
            # 1. 기본 네트워크 메트릭스
            self.network_metrics['diameter'] = nx.diameter(G)
            self.network_metrics['radius'] = nx.radius(G)
            self.network_metrics['density'] = nx.density(G)
            
            # 2. 거리별 노드 분포 분석
            distances = nx.single_source_shortest_path_length(G, self.gate_node)
            distance_distribution = defaultdict(list)
            for node, dist in distances.items():
                distance_distribution[dist].append(node)
            
            self.network_metrics['distance_distribution'] = dict(distance_distribution)
            self.network_metrics['max_distance'] = max(distances.values())
            
            # 3. 네트워크 형태 분류
            self.network_metrics['network_type'] = self._classify_network_type()
            
            # 4. 병목 지역 식별
            articulation_points = list(nx.articulation_points(G))
            self.network_metrics['bottlenecks'] = articulation_points
            
            # 5. 말단 지역 식별
            terminal_regions = self._identify_terminal_regions()
            self.network_metrics['terminal_regions'] = terminal_regions
            
            # 6. 중심성 기반 중요도 분석
            centrality_metrics = self._calculate_centrality_metrics()
            self.network_metrics.update(centrality_metrics)
        
        def _classify_network_type(self):
            """네트워크 형태 분류"""
            G = self.graph
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            avg_degree = 2 * n_edges / n_nodes
            
            if n_edges <= n_nodes - 1:
                return 'tree'
            elif avg_degree < 2.5:
                return 'linear'
            elif 2.5 <= avg_degree <= 4.0:
                return 'grid'
            else:
                return 'complex'
        
        def _identify_terminal_regions(self):
            """말단 지역 식별"""
            G = self.graph
            terminal_regions = {}
            
            leaf_nodes = [n for n in G.nodes() if G.degree(n) == 1 and n != self.gate_node]
            
            for leaf in leaf_nodes:
                region_nodes = set([leaf])
                for neighbor in G.neighbors(leaf):
                    region_nodes.add(neighbor)
                    for second_neighbor in G.neighbors(neighbor):
                        if G.degree(second_neighbor) <= 3:
                            region_nodes.add(second_neighbor)
                
                terminal_regions[leaf] = list(region_nodes)
            
            return terminal_regions
        
        def _calculate_centrality_metrics(self):
            """중심성 지표 계산"""
            G = self.graph
            
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            
            distances = nx.single_source_shortest_path_length(G, self.gate_node)
            max_dist = max(distances.values())
            gate_accessibility = {n: 1 - (d / max_dist) for n, d in distances.items()}
            
            return {
                'betweenness_centrality': betweenness,
                'closeness_centrality': closeness,
                'gate_accessibility': gate_accessibility
            }

    class AdaptiveWeightSystem:
        """수요 패턴과 네트워크 구조 기반 적응형 가중치 시스템"""
        
        def __init__(self, network_analyzer):
            self.network_analyzer = network_analyzer
            self.base_weights = self._initialize_base_weights()
        
        def _initialize_base_weights(self):
            """네트워크 타입별 기본 가중치 설정"""
            network_type = self.network_analyzer.network_metrics['network_type']
            
            base_weights = {
                'tree': {
                    'criticality_weight': 20,
                    'connectivity_weight': -25,
                    'distance_preference': 'balanced',
                    'terminal_bonus': 15
                },
                'linear': {
                    'criticality_weight': 25,
                    'connectivity_weight': -20,
                    'distance_preference': 'far_first',
                    'terminal_bonus': 20
                },
                'grid': {
                    'criticality_weight': 12,
                    'connectivity_weight': -35,
                    'distance_preference': 'adaptive',
                    'terminal_bonus': 10
                },
                'complex': {
                    'criticality_weight': 8,
                    'connectivity_weight': -40,
                    'distance_preference': 'adaptive',
                    'terminal_bonus': 5
                }
            }
            
            return base_weights.get(network_type, base_weights['grid'])
        
        def calculate_adaptive_weights(self, current_situation):
            """현재 상황에 맞는 적응형 가중치 계산"""
            base = self.base_weights.copy()
            
            remaining_space = current_situation['remaining_space']
            remaining_demands = current_situation['remaining_demands']
            is_next_port = current_situation.get('is_next_port', False)
            demand_density = current_situation.get('demand_density', 0.5)
            
            # 공간 압박도에 따른 조정
            space_pressure = remaining_demands / max(remaining_space, 1)
            
            if space_pressure > 1.5:
                base['criticality_weight'] *= 1.8
                base['connectivity_weight'] *= 0.7
            elif space_pressure > 1.2:
                base['criticality_weight'] *= 1.4
                base['connectivity_weight'] *= 0.85
            
            # 다음 항구 수요에 대한 특별 처리
            if is_next_port:
                base['criticality_weight'] *= 0.3
                base['connectivity_weight'] = -10
                base['terminal_bonus'] = 0
                return base
            
            # 말단 지역 활용도 조정
            terminal_utilization = self._calculate_terminal_utilization(current_situation)
            if terminal_utilization < 0.3:
                base['terminal_bonus'] *= 2.0
            
            # 수요 밀도에 따른 조정
            if demand_density > 0.8:
                base['connectivity_weight'] *= 1.3
            elif demand_density < 0.3:
                base['terminal_bonus'] *= 1.5
            
            return base
        
        def _calculate_terminal_utilization(self, situation):
            """말단 지역 활용도 계산"""
            terminal_regions = self.network_analyzer.network_metrics['terminal_regions']
            used_nodes = situation.get('used_nodes', set())
            
            total_terminal_nodes = sum(len(region) for region in terminal_regions.values())
            if total_terminal_nodes == 0:
                return 1.0
            
            used_terminal_nodes = 0
            for region in terminal_regions.values():
                used_terminal_nodes += len([n for n in region if n in used_nodes])
            
            return used_terminal_nodes / total_terminal_nodes

    def calculate_distance_score_enhanced(node_distance, max_distance, network_type, is_next_port):
        """향상된 거리 점수 계산"""
        if is_next_port:
            # 다음 항구: 가까운 거리 선호
            if 1 <= node_distance <= 3:
                return 0  # 최고 점수
            elif 4 <= node_distance <= 5:
                return 3
            else:
                return 7
        
        # 네트워크 타입별 거리 선호도
        if network_type == 'linear':
            # Linear: 원거리부터 채우기
            mid_point = max_distance // 2
            if node_distance >= mid_point + 1:
                return 0
            elif node_distance >= mid_point - 1:
                return 2
            else:
                return 6
        
        elif network_type == 'tree':
            # Tree: 균형 잡힌 배치
            mid_point = max_distance // 2
            if max(1, mid_point - 1) <= node_distance <= mid_point + 1:
                return 0
            elif node_distance >= mid_point + 2:
                return 2
            else:
                return 4
        
        else:  # grid, complex
            mid_point = max_distance // 2
            optimal_range = max(2, mid_point // 2)
            distance_from_optimal = abs(node_distance - (mid_point - optimal_range))
            return min(distance_from_optimal, 8)

    def calculate_node_quality_enhanced_v2(node, current_cluster, available_nodes, graph, critical_nodes, 
                                         network_analyzer, adaptive_weights, current_situation, gate_node=0):
        """완전히 개선된 노드 품질 계산 함수"""
        
        # 적응형 가중치 계산
        weights = adaptive_weights.calculate_adaptive_weights(current_situation)
        
        # 1. 중요도 페널티
        criticality_penalty = critical_nodes.get(node, 0) * weights['criticality_weight']
        
        # 2. 연결성 보너스
        connectivity_bonus = 0
        for cluster_node in current_cluster:
            if graph.has_edge(node, cluster_node):
                connectivity_bonus += 1
        connectivity_score = connectivity_bonus * weights['connectivity_weight']
        
        # 3. 향상된 거리 점수
        try:
            node_distance = nx.shortest_path_length(graph, gate_node, node)
        except:
            node_distance = float('inf')
            
        max_distance = network_analyzer.network_metrics['max_distance']
        network_type = network_analyzer.network_metrics['network_type']
        is_next_port = current_situation.get('is_next_port', False)
        
        distance_score = calculate_distance_score_enhanced(node_distance, max_distance, network_type, is_next_port)
        
        # 4. 말단 지역 보너스
        terminal_bonus = 0
        terminal_regions = network_analyzer.network_metrics['terminal_regions']
        for region_nodes in terminal_regions.values():
            if node in region_nodes:
                terminal_bonus = weights['terminal_bonus']
                break
        
        # 5. 네트워크 중심성 고려
        centrality_metrics = network_analyzer.network_metrics
        betweenness_penalty = centrality_metrics['betweenness_centrality'].get(node, 0) * 10
        
        # 6. 접근성 영향 (최적화된 버전)
        accessibility_penalty = 0
        temp_unavailable = current_cluster | {node}
        
        test_nodes = [n for n in available_nodes if n not in temp_unavailable]
        if len(test_nodes) > 10:
            test_nodes = random.sample(test_nodes, 10)
        
        for test_node in test_nodes:
            try:
                temp_graph = graph.copy()
                temp_graph.remove_nodes_from(temp_unavailable)
                if not nx.has_path(temp_graph, gate_node, test_node):
                    accessibility_penalty += 1
            except:
                accessibility_penalty += 1
        
        # 최종 점수 계산 (낮을수록 좋음)
        final_score = (
            criticality_penalty + 
            accessibility_penalty * 5 + 
            connectivity_score + 
            distance_score + 
            betweenness_penalty - 
            terminal_bonus
        )
        
        return final_score

    #------------- begin of improved algorithm code --------------#

    def get_critical_path_nodes(graph, target_nodes, gate_node=0):
        """중요 경로상의 노드들을 식별"""
        critical_nodes = {}
        
        for target in target_nodes:
            try:
                path = nx.shortest_path(graph, gate_node, target)
                for node in path[:-1]:
                    if node not in critical_nodes:
                        critical_nodes[node] = 0
                    critical_nodes[node] += 1
            except nx.NetworkXNoPath:
                continue
        
        critical_threshold = max(2, len(target_nodes) * 0.1)
        return {node: count for node, count in critical_nodes.items() if count >= critical_threshold}

    def find_path_aware_cluster_enhanced(start_nodes, target_size, available_nodes, graph, critical_nodes, 
                                       network_analyzer, adaptive_weights, current_situation, gate_node=0):
        """향상된 경로 인식 클러스터 형성"""
        if target_size <= 0:
            return []
        
        best_cluster = []
        best_score = float('inf')
        
        # 시작 노드들을 중요도에 따라 정렬
        start_candidates = []
        for node in start_nodes:
            if node in available_nodes:
                criticality = critical_nodes.get(node, 0)
                try:
                    distance = nx.shortest_path_length(graph, gate_node, node)
                except:
                    distance = float('inf')
                start_candidates.append((criticality, distance, node))
        
        start_candidates.sort(key=lambda x: (x[0], -x[1]))
        
        # 여러 시작점에서 클러스터 형성 시도
        for criticality, distance, start_node in start_candidates[:min(3, len(start_candidates))]:
            cluster = build_cluster_from_start_enhanced(start_node, target_size, available_nodes, 
                                                      graph, critical_nodes, network_analyzer, 
                                                      adaptive_weights, current_situation, gate_node)
            
            if len(cluster) >= target_size:
                cluster_score = evaluate_cluster_quality_enhanced(cluster, graph, critical_nodes, 
                                                                available_nodes, network_analyzer, 
                                                                adaptive_weights, current_situation, gate_node)
                
                if cluster_score < best_score:
                    best_score = cluster_score
                    best_cluster = cluster
        
        return best_cluster[:target_size] if best_cluster else []
    
    def build_cluster_from_start_enhanced(start_node, target_size, available_nodes, graph, critical_nodes, 
                                        network_analyzer, adaptive_weights, current_situation, gate_node):
        """향상된 클러스터 구축"""
        cluster = {start_node}
        queue = [start_node]
        
        while queue and len(cluster) < target_size:
            current = queue.pop(0)
            
            neighbor_candidates = []
            for neighbor in graph.neighbors(current):
                if neighbor in available_nodes and neighbor not in cluster:
                    quality_score = calculate_node_quality_enhanced_v2(
                        neighbor, cluster, available_nodes, graph, critical_nodes,
                        network_analyzer, adaptive_weights, current_situation, gate_node
                    )
                    neighbor_candidates.append((quality_score, neighbor))
            
            neighbor_candidates.sort(key=lambda x: x[0])
            
            for quality_score, neighbor in neighbor_candidates:
                if len(cluster) < target_size:
                    cluster.add(neighbor)
                    queue.append(neighbor)
                else:
                    break
        
        return list(cluster)
    
    def evaluate_cluster_quality_enhanced(cluster, graph, critical_nodes, available_nodes, 
                                        network_analyzer, adaptive_weights, current_situation, gate_node):
        """향상된 클러스터 품질 평가"""
        weights = adaptive_weights.calculate_adaptive_weights(current_situation)
        
        # 내부 연결성 점수
        internal_connections = 0
        for i, node1 in enumerate(cluster):
            for node2 in cluster[i+1:]:
                if graph.has_edge(node1, node2):
                    internal_connections += 1
        
        max_possible_connections = len(cluster) * (len(cluster) - 1) / 2
        connectivity_score = internal_connections / max(1, max_possible_connections)
        
        # 중요 노드 사용 페널티
        critical_penalty = sum(critical_nodes.get(node, 0) for node in cluster) * weights['criticality_weight']
        
        # 접근성 영향 페널티
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
        
        return (critical_penalty + 
                blocked_count * 20 + 
                connectivity_score * weights['connectivity_weight'] + 
                cluster_span * 5)

    def find_connected_cluster(start_nodes, target_size, available_nodes, graph):
        """fallback용 연결된 클러스터 찾기"""
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

    def loading_heuristic(p, node_allocations, rehandling_demands, node_priorities):
        print(f'Starting loading phase...')

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
            # IMPROVED: 고도화된 경로 인식 클러스터 배치 + 네트워크 구조 분석
            print("=== Applying enhanced path-aware cluster allocation strategy ===")
            
            # 네트워크 분석 시스템 초기화 (한 번만)
            network_analyzer = NetworkAnalyzer(G, 0)
            adaptive_weights = AdaptiveWeightSystem(network_analyzer)
            
            print(f"Network analysis: type={network_analyzer.network_metrics['network_type']}, "
                  f"diameter={network_analyzer.network_metrics['diameter']}, "
                  f"terminal_regions={len(network_analyzer.network_metrics['terminal_regions'])}")
            
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
            print(f"Destination priority (latest first): {sorted_destinations}")
            
            # 1.5. 다음 항구(p+1) 수요 분리 및 우선 처리 계획
            next_port = p + 1
            next_port_demands = demands_by_destination.get(next_port, [])
            
            if next_port_demands:
                print(f"*** Next port priority: {len(next_port_demands)} containers for port {next_port} will be placed near gate ***")
                # 다음 항구 수요를 제외한 나머지를 먼저 처리
                priority_destinations = [dest for dest in sorted_destinations if dest != next_port]
                priority_destinations.append(next_port)  # 다음 항구를 마지막에 추가
            else:
                priority_destinations = sorted_destinations
            
            # 2. 중요 경로 노드들 식별
            critical_nodes = get_critical_path_nodes(G, reachable_nodes, 0)
            print(f"Critical path nodes identified: {len(critical_nodes)} nodes")
            if critical_nodes:
                top_critical = sorted(critical_nodes.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"Most critical nodes: {top_critical}")
            
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
            
            print(f"Available nodes by distance (first 3 layers): {dict(list(reachable_nodes_by_distance.items())[:3])}")
            
            # 4. 다중 클러스터 기반 동적 우선순위 할당
            loading_assignments = []
            used_nodes = set()
            allocation_history = []  # 할당 기록 저장
            
            # *** 다중 클러스터 전략 초기화 ***
            multi_cluster_strategy = MultiClusterStrategy(network_analyzer, adaptive_weights)
            total_clusters_formed = 0
            
            for dest_idx, dest in enumerate(priority_destinations):
                dest_demands = demands_by_destination[dest]
                total_demand_size = len(dest_demands)
                
                # 다음 항구 수요인지 확인
                is_next_port = (dest == next_port)
                
                # 현재 상황에 맞게 중요도 가중치 동적 조정
                remaining_demands = sum(len(demands_by_destination[d]) for d in priority_destinations[dest_idx:])
                remaining_space = len([n for n in reachable_nodes if n not in used_nodes])
                
                print(f"\n  === Destination {dest}: {total_demand_size} containers ===")
                print(f"    Remaining space: {remaining_space}, remaining demands: {remaining_demands}")
                
                # *** 다중 클러스터 전략 적용 ***
                cluster_count = multi_cluster_strategy.calculate_optimal_cluster_count(
                    total_demand_size, len(G.nodes()), remaining_space, is_next_port
                )
                
                print(f"    Optimal cluster count: {cluster_count}")
                
                if cluster_count > 1:
                    print(f"    *** Multi-cluster strategy: splitting {total_demand_size} demands into {cluster_count} clusters ***")
                elif is_next_port:
                    print(f"    *** Next Port {dest}: {total_demand_size} containers - prioritizing near-gate placement ***")
                else:
                    print(f"    Single cluster strategy: {total_demand_size} containers - forming path-aware cluster")
                
                # 수요를 클러스터별로 분배
                demand_clusters = multi_cluster_strategy.distribute_demands_to_clusters(dest_demands, cluster_count)
                print(f"    Cluster sizes: {[len(cluster) for cluster in demand_clusters]}")
                
                # 각 클러스터별로 할당 수행
                for cluster_idx, cluster_demands in enumerate(demand_clusters):
                    cluster_size = len(cluster_demands)
                    
                    if cluster_count > 1:
                        print(f"    --- Cluster {cluster_idx + 1}/{cluster_count}: {cluster_size} containers ---")
                    
                    # 현재 상황 업데이트 (클러스터별로)
                    current_remaining_space = len([n for n in reachable_nodes if n not in used_nodes])
                    current_remaining_demands = sum(len(demands_by_destination[d]) for d in priority_destinations[dest_idx:]) - sum(len(dc) for dc in demand_clusters[:cluster_idx])
                    
                    # 클러스터별 시작점 선택 전략
                    if is_next_port:
                        # 다음 항구: 모든 클러스터를 가까운 곳에
                        distance_order = sorted(reachable_nodes_by_distance.keys())  # 가까운 거리부터
                        if cluster_count == 1:
                            print(f"    Next port strategy: placing near gate (distances: {distance_order[:3]})")
                        else:
                            print(f"      Next port cluster {cluster_idx + 1}: targeting near gate")
                    else:
                        # 일반 수요: 클러스터별로 다른 거리 영역 활용
                        if cluster_count > 1:
                            # 여러 클러스터인 경우: 거리별로 분산 배치
                            available_distances = sorted(reachable_nodes_by_distance.keys(), reverse=True)
                            if cluster_idx < len(available_distances):
                                # 각 클러스터를 다른 거리 영역에 우선 배치
                                preferred_distance = available_distances[cluster_idx % len(available_distances)]
                                distance_order = [preferred_distance] + [d for d in available_distances if d != preferred_distance]
                                print(f"      Cluster {cluster_idx + 1} targeting distance layer {preferred_distance}")
                            else:
                                distance_order = available_distances
                        else:
                            # 단일 클러스터: 기존 방식 (먼 거리부터)
                            distance_order = sorted(reachable_nodes_by_distance.keys(), reverse=True)
                    
                    # 클러스터 형성을 위한 시작점 찾기
                    cluster_nodes = []
                    
                    for distance in distance_order:
                        available_start_nodes = [n for n in reachable_nodes_by_distance[distance] 
                                               if n not in used_nodes]
                        
                        if available_start_nodes:
                            # 클러스터 형성 방식도 다음 항구에 따라 조정
                            remaining_available = [n for n in reachable_nodes if n not in used_nodes]
                            
                            # 현재 상황 정보 구성
                            current_situation = {
                                'remaining_space': current_remaining_space,
                                'remaining_demands': current_remaining_demands,
                                'destination': dest,
                                'is_next_port': is_next_port,
                                'used_nodes': used_nodes,
                                'available_nodes': remaining_available,
                                'critical_nodes': critical_nodes,
                                'demand_density': cluster_size / max(current_remaining_space, 1),
                                'cluster_index': cluster_idx,
                                'total_clusters': cluster_count
                            }
                            
                            if is_next_port:
                                # 다음 항구: 간단한 연결된 클러스터 (빠른 접근 우선)
                                cluster_nodes = find_connected_cluster(
                                    available_start_nodes[:1],  # 첫 번째 노드부터 시작
                                    cluster_size, 
                                    remaining_available, 
                                    G
                                )
                                if len(cluster_nodes) >= cluster_size:
                                    if cluster_count == 1:
                                        print(f"    Next port cluster: simple connected cluster from distance {distance}")
                                    else:
                                        print(f"      Cluster formed at distance {distance}")
                                    break
                            else:
                                # 일반 수요: 고도화된 경로 인식 클러스터 형성
                                cluster_nodes = find_path_aware_cluster_enhanced(
                                    available_start_nodes[:2 if cluster_count > 1 else 3],  # 다중 클러스터시 후보 수 줄임
                                    cluster_size, 
                                    remaining_available, 
                                    G, 
                                    critical_nodes,
                                    network_analyzer,
                                    adaptive_weights,
                                    current_situation
                                )
                                if len(cluster_nodes) >= cluster_size:
                                    if cluster_count == 1:
                                        print(f"    Single cluster formed at distance {distance}")
                                    else:
                                        print(f"      Cluster {cluster_idx + 1} formed at distance {distance}")
                                    break
                    
                    # 클러스터를 찾지 못한 경우, fallback 방식 사용
                    if len(cluster_nodes) < cluster_size:
                        if cluster_count > 1:
                            print(f"      Cluster {cluster_idx + 1} primary formation failed, trying fallback...")
                        else:
                            print(f"    Primary cluster formation failed, trying fallback methods...")
                        
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
                            if cluster_count == 1:
                                print(f"    Next port fallback: Sequential allocation from gate")
                            else:
                                print(f"      Cluster {cluster_idx + 1} fallback: Sequential allocation from gate")
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
                                        if cluster_count == 1:
                                            print(f"    Fallback 1 successful: Connected cluster formed")
                                        else:
                                            print(f"      Cluster {cluster_idx + 1} fallback successful")
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
                            if cluster_count == 1:
                                print(f"    Next port final fallback: Distance-based sequential allocation")
                            else:
                                print(f"      Cluster {cluster_idx + 1} final fallback: Distance-based allocation")
                        else:
                            # 일반 수요: 중요도에 따라 정렬
                            remaining_with_criticality = []
                            for n in remaining_available:
                                criticality = critical_nodes.get(n, 0)
                                remaining_with_criticality.append((criticality, n))
                            remaining_with_criticality.sort(key=lambda x: x[0])  # 덜 중요한 것부터
                            cluster_nodes = [n for _, n in remaining_with_criticality[:cluster_size]]
                            if cluster_count == 1:
                                print(f"    Fallback 2: Sequential allocation with criticality consideration")
                            else:
                                print(f"      Cluster {cluster_idx + 1} fallback: Criticality-based allocation")
                    
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
                        if assigned_count < len(cluster_demands) and node not in used_nodes:  # 중복 할당 방지
                            demand_k = cluster_demands[assigned_count]
                            loading_assignments.append((node, demand_k))
                            used_nodes.add(node)
                            allocation_history.append((dest, node, criticality, distance))
                            
                            # 로그 출력 조건 조정
                            if cluster_count == 1 and assigned_count < 3:  # 단일 클러스터는 처음 3개
                                critical_info = f"(critical: {criticality})" if criticality > 0 else ""
                                print(f"    Demand {demand_k} -> Node {node} (dist {distance}, pos {assigned_count+1}) {critical_info}")
                            elif cluster_count > 1 and assigned_count < 2:  # 다중 클러스터는 처음 2개만
                                critical_info = f"(critical: {criticality})" if criticality > 0 else ""
                                print(f"      Demand {demand_k} -> Node {node} (dist {distance}) {critical_info}")
                            
                            assigned_count += 1
                        elif node in used_nodes:
                            print(f"    Warning: Node {node} already used, skipping assignment")
                    
                    # 할당이 부족한 경우 추가 노드 찾기
                    if assigned_count < len(cluster_demands):
                        remaining_demands_count = len(cluster_demands) - assigned_count
                        if cluster_count == 1:
                            print(f"    Need {remaining_demands_count} more nodes for destination {dest}")
                        else:
                            print(f"      Cluster {cluster_idx + 1} needs {remaining_demands_count} more nodes")
                        
                        # 사용 가능한 추가 노드 찾기
                        additional_nodes = []
                        for n in reachable_nodes:
                            if n not in used_nodes and len(additional_nodes) < remaining_demands_count:
                                distance = len(shortest_paths[n][0]) - 1 if shortest_paths[n] else float('inf')
                                criticality = critical_nodes.get(n, 0)
                                additional_nodes.append((distance, criticality, n))
                        
                        # 거리와 중요도로 정렬
                        additional_nodes.sort(key=lambda x: (x[0], x[1]))
                        
                        for distance, criticality, node in additional_nodes:
                            if assigned_count < len(cluster_demands):
                                demand_k = cluster_demands[assigned_count]
                                loading_assignments.append((node, demand_k))
                                used_nodes.add(node)
                                allocation_history.append((dest, node, criticality, distance))
                                if cluster_count == 1:
                                    print(f"    Additional: Demand {demand_k} -> Node {node} (dist {distance}) {f'(critical: {criticality})' if criticality > 0 else ''}")
                                else:
                                    print(f"      Additional: Demand {demand_k} -> Node {node} (dist {distance}) {f'(critical: {criticality})' if criticality > 0 else ''}")
                                assigned_count += 1
                    
                    # 클러스터 완료 로그
                    cluster_critical_count = sum(1 for _, c, _ in cluster_nodes_with_distance if c > 0)
                    if is_next_port:
                        avg_distance = sum(d for d, _, _ in cluster_nodes_with_distance) / len(cluster_nodes_with_distance) if cluster_nodes_with_distance else 0
                        if cluster_count == 1:
                            print(f"    *** Next port cluster formed: {len(cluster_nodes)} nodes, avg distance: {avg_distance:.1f}, {cluster_critical_count} critical nodes used ***")
                        else:
                            print(f"    *** Next port cluster {cluster_idx + 1} formed: {len(cluster_nodes)} nodes, avg distance: {avg_distance:.1f} ***")
                    else:
                        if cluster_count == 1:
                            print(f"    Cluster formed: {len(cluster_nodes)} nodes, {cluster_critical_count} critical nodes used")
                        else:
                            print(f"      Cluster {cluster_idx + 1} formed: {len(cluster_nodes)} nodes, {cluster_critical_count} critical nodes used")
                    
                    total_clusters_formed += 1
                
                # 목적지별 완료 로그
                print(f"    Destination {dest} completed: {cluster_count} cluster{'s' if cluster_count > 1 else ''} formed")
                
                # 할당 품질 모니터링
                if dest_idx % 3 == 0:  # 3번째마다 품질 체크
                    total_critical_used = sum(1 for _, _, c, _ in allocation_history if c > 0)
                    avg_distance = sum(d for _, _, _, d in allocation_history) / len(allocation_history) if allocation_history else 0
                    print(f"    Quality check: {total_critical_used} critical nodes used, avg distance: {avg_distance:.1f}")

            # 다중 클러스터 전략 요약
            print(f"\n=== Multi-cluster allocation summary ===")
            print(f"Total clusters formed: {total_clusters_formed}")
            print(f"Average cluster size: {len(loading_assignments) / max(total_clusters_formed, 1):.1f}")
            
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
                            print(f"    Fixed: Demand {demand_k} reassigned to Node {n}")
                            break
                
                loading_assignments = unique_assignments
                used_nodes = used_in_final
                print(f"    Fixed: {len(missing_demands)} duplicate assignments resolved")
            else:
                print("✓ No duplicate assignments found")

            assert(len(loading_assignments) == total_loading_demands)

            # 5. 할당 후 접근성 검증 및 적응적 조정
            remaining_nodes = [n for n in reachable_nodes if n not in used_nodes]
            blocked_nodes = []
            for node in remaining_nodes:
                temp_graph = G.copy()
                temp_graph.remove_nodes_from(used_nodes)
                if not nx.has_path(temp_graph, 0, node):
                    blocked_nodes.append(node)
            
            if blocked_nodes:
                print(f"    Warning: {len(blocked_nodes)} nodes became unreachable after allocation")
                print(f"    Blocked nodes: {blocked_nodes[:10]}{'...' if len(blocked_nodes) > 10 else ''}")
                
                # 적응적 조정: 중요 노드가 불필요하게 사용되었는지 확인하고 조정
                if len(blocked_nodes) > len(reachable_nodes) * 0.1:  # 10% 이상 막혔으면 조정
                    print(f"    Attempting adaptive adjustment...")
                    
                    # 현재 할당에서 중요도가 높으면서 교체 가능한 노드 찾기
                    adjustable_assignments = []
                    for idx, (node, demand_k) in enumerate(loading_assignments):
                        node_criticality = critical_nodes.get(node, 0)
                        if node_criticality > 0:
                            # 이 노드를 다른 덜 중요한 노드로 교체할 수 있는지 확인
                            alternative_nodes = [n for n in remaining_nodes + blocked_nodes 
                                               if critical_nodes.get(n, 0) < node_criticality]
                            if alternative_nodes:
                                adjustable_assignments.append((idx, node, demand_k, node_criticality, alternative_nodes))
                    
                    # 가장 중요한 노드부터 교체 시도
                    adjustable_assignments.sort(key=lambda x: x[3], reverse=True)
                    
                    adjustments_made = 0
                    already_used_alternatives = set()  # 이미 사용된 대체 노드 추적
                    
                    for idx, old_node, demand_k, criticality, alternatives in adjustable_assignments[:3]:  # 최대 3개만 조정
                        # 가장 좋은 대안 선택 (이미 사용된 것 제외)
                        best_alternative = None
                        best_score = float('inf')
                        
                        available_alternatives = [alt for alt in alternatives[:10] 
                                                if alt not in already_used_alternatives and alt not in used_nodes]
                        
                        for alt_node in available_alternatives:  # 사용 가능한 대안만 검토
                            # 이 교체가 전체적으로 도움이 되는지 평가
                            temp_used = (used_nodes - {old_node}) | {alt_node}
                            temp_remaining = [n for n in reachable_nodes if n not in temp_used]
                            temp_blocked = 0
                            
                            for test_node in temp_remaining[:10]:  # 샘플링으로 빠르게 평가
                                temp_graph = G.copy()
                                temp_graph.remove_nodes_from(temp_used)
                                try:
                                    if not nx.has_path(temp_graph, 0, test_node):
                                        temp_blocked += 1
                                except:
                                    temp_blocked += 1
                            
                            score = temp_blocked * 10 + critical_nodes.get(alt_node, 0)
                            if score < best_score:
                                best_score = score
                                best_alternative = alt_node
                        
                        if best_alternative and best_alternative not in used_nodes:
                            # 교체 실행
                            loading_assignments[idx] = (best_alternative, demand_k)
                            used_nodes.remove(old_node)
                            used_nodes.add(best_alternative)
                            already_used_alternatives.add(best_alternative)  # 사용된 대체 노드 기록
                            adjustments_made += 1
                            print(f"      Adjusted: Node {old_node} (critical: {criticality}) -> Node {best_alternative} (critical: {critical_nodes.get(best_alternative, 0)})")
                            
                            if adjustments_made >= 3:  # 최대 3개만 조정
                                break
                        else:
                            print(f"      Skipped adjustment for Node {old_node}: no suitable alternative available")
                    
                    if adjustments_made > 0:
                        # 조정 후 재검증
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

            # 6. 최종 할당 품질 리포트
            total_critical_used = sum(1 for node, _ in loading_assignments if critical_nodes.get(node, 0) > 0)
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
            print(f"      Network type: {network_analyzer.network_metrics['network_type']}")
            print(f"      Critical nodes used: {total_critical_used}/{len(loading_assignments)} ({total_critical_used/len(loading_assignments)*100:.1f}%)")
            print(f"      Average distance: {avg_distance:.1f}")
            print(f"      Average cluster cohesion: {avg_cohesion:.2f}")
            print(f"      Nodes blocked: {len(blocked_nodes) if 'blocked_nodes' in locals() else 0}")
            print(f"      Terminal regions utilized: {len([r for r in network_analyzer.network_metrics['terminal_regions'].values() if any(n in used_nodes for n in r)])}/{len(network_analyzer.network_metrics['terminal_regions'])}")

            # 7. 실제 할당 및 경로 생성
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

        print(f'Unloading phase completed with {len(route_list)} routes including {len(rehandling_demands)} rehandling routes!')
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
            # Loading heuristic with enhanced path-aware cluster placement
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