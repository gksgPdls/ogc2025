# myalgorithm.py
# This is a template for the custom algorithm.
# Last updated: 2025/07/19

import time
import util
import networkx as nx
from itertools import islice
import numpy as np
import traceback

# 만든 파일 이름 import 하기
import vertical_v7
import multi_cluster_v1
import vertical_v5
import vertical_v3_next_v1_multitry_v2
import vertical_v8

def algorithm(prob_info, timelimit=60):
    start_time = time.time()
    
    # 점수 제일 낮은 솔루션을 반환
    def best_solution_finder(solution_list):
        if not solution_list:
            return None
            
        best_solution = solution_list[0]
        for i in range(1, len(solution_list)):
            if best_solution['obj'] > solution_list[i]['obj']:
                best_solution = solution_list[i]
        return best_solution['solution']
    
    # 안전한 솔루션 실행 함수
    def safe_run_algorithm(prob_info, function_name, remaining_time, function_index):
        try:
            print(f"  🚀 Starting execution...")
            start_exec = time.time()
            temp_solution = function_name(prob_info, max(3, remaining_time))  # 최소 3초는 주기
            end_exec = time.time()
            exec_time = end_exec - start_exec
            
            print(f"  ⏱️ Execution completed in {exec_time:.2f}s")
            score = util.check_feasibility(prob_info, temp_solution)
            obj_score = score.get('obj', float('inf'))
            print(f"  📊 Score: {obj_score}")
            return {'solution': temp_solution, 'obj': obj_score, 'success': True, 'exec_time': exec_time}
        except Exception as e:
            end_exec = time.time()
            exec_time = end_exec - start_exec if 'start_exec' in locals() else 0
            print(f"  💥 Failed after {exec_time:.2f}s: {str(e)}")
            # 에러 타입별 상세 정보
            if "timeout" in str(e).lower() or "time" in str(e).lower():
                print(f"  🕐 Likely timeout error")
            elif "memory" in str(e).lower():
                print(f"  💾 Likely memory error") 
            elif "index" in str(e).lower():
                print(f"  📋 Likely index/data error")
            else:
                print(f"  ❓ Other error type")
            return {'solution': None, 'obj': float('inf'), 'success': False, 'error': str(e), 'exec_time': exec_time}
    
    # 기본 fallback 솔루션 생성 (간단한 솔루션)
    def create_fallback_solution(prob_info):
        try:
            print("Creating fallback solution...")
            P = prob_info['P']
            # 가장 간단한 솔루션: 모든 점을 첫 번째 클러스터에 할당
            fallback = [0] * P
            score = util.check_feasibility(prob_info, fallback)
            obj_score = score.get('obj', float('inf'))
            print(f"Fallback solution created with score: {obj_score}")
            return {'solution': fallback, 'obj': obj_score, 'success': True}
        except Exception as e:
            print(f"Even fallback solution failed: {str(e)}")
            # 최후의 수단: 빈 리스트 반환
            return {'solution': [], 'obj': float('inf'), 'success': False}
    
    P = prob_info['P']
    solution_list = []
    
    # 함수들 정의 (우선순위 순서로 배치 - 성능 좋은 것부터)
    function_list = [
        vertical_v5.algorithm,
        vertical_v7.algorithm,
        vertical_v3_next_v1_multitry_v2.algorithm,
        multi_cluster_v1.algorithm,
        vertical_v8.algorithm
    ]
    
    # 먼저 fallback 솔루션 생성
    fallback_result = create_fallback_solution(prob_info)
    if fallback_result['success']:
        solution_list.append(fallback_result)
    
    print(f"Starting algorithm execution with {timelimit} seconds timelimit")
    
    # 각 함수를 안전하게 실행
    for i, function_name in enumerate(function_list):
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = timelimit - elapsed_time
        
        print(f"\n=== Algorithm {i+1}/{len(function_list)}: {function_name.__name__} ===")
        print(f"Time elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s")
        
        # 더 관대한 시간 체크: 5초만 남겨두기 (10초 → 5초)
        if remaining_time <= 5:
            print(f"Time running low ({remaining_time:.2f}s remaining), stopping execution")
            break
        
        # 각 알고리즘에 더 많은 시간 할당
        remaining_functions = len(function_list) - i
        # 최소 3초, 남은 시간의 절반을 현재 함수에 할당하거나 균등분배 중 큰 값
        min_time_per_function = max(3, remaining_time * 0.1)  # 남은 시간의 10% 또는 최소 3초
        equal_distribution = (remaining_time - 2) / remaining_functions  # 마지막 2초만 여유
        allocated_time = max(min_time_per_function, equal_distribution)
        
        print(f"Remaining functions: {remaining_functions}")
        print(f"Allocated time for this algorithm: {allocated_time:.2f}s")
        
        # 안전하게 알고리즘 실행
        algo_start = time.time()
        result = safe_run_algorithm(prob_info, function_name, allocated_time, i)
        algo_end = time.time()
        actual_time = algo_end - algo_start
        
        print(f"Algorithm took {actual_time:.2f}s (allocated: {allocated_time:.2f}s)")
        
        if result['success'] and result['solution'] is not None:
            solution_list.append(result)
            print(f"✅ Successfully added solution with score: {result['obj']}")
        else:
            print(f"❌ Algorithm {i+1} failed: {result.get('error', 'Unknown error')}")
        
        # 더 관대한 시간 체크 (3초 여유)
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= timelimit - 3:
            print(f"⏰ Approaching time limit (used {elapsed_time:.2f}s/{timelimit}s), stopping execution")
            break
    
    print(f"\nTotal solutions collected: {len(solution_list)}")
    for i, sol in enumerate(solution_list):
        print(f"Solution {i+1}: score = {sol['obj']}")
    
    # 최적 솔루션 선택
    if solution_list:
        best_solution = best_solution_finder(solution_list)
        if best_solution is not None:
            print(f"Returning best solution")
            return best_solution
    
    # 모든 솔루션이 실패한 경우 최후의 수단
    print("All algorithms failed, returning emergency fallback")
    P = prob_info['P']
    return [0] * P  # 모든 점을 첫 번째 클러스터에 할당


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