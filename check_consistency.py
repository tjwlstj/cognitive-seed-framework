#!/usr/bin/env python3
"""
프로젝트 일관성 체크 스크립트

누락된 파일, 불일치하는 정보, 오타 등을 검사합니다.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent

# 예상되는 시드 목록
EXPECTED_SEEDS = {
    'atomic': ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08'],
    'molecular': ['m01', 'm02', 'm03', 'm04', 'm05', 'm06', 'm07', 'm08'],
}

# 시드 이름 매핑
SEED_NAMES = {
    'a01': 'Edge Detector',
    'a02': 'Symmetry Detector',
    'a03': 'Recurrence Spotter',
    'a04': 'Contrast Amplifier',
    'a05': 'Grouping Nucleus',
    'a06': 'Sequence Tracker',
    'a07': 'Scale Normalizer',
    'a08': 'Binary Comparator',
    'm01': 'Hierarchy Builder',
    'm02': 'Causality Detector',
    'm03': 'Pattern Completer',
    'm04': 'Spatial Transformer',
    'm05': 'Concept Crystallizer',
    'm06': 'Context Integrator',
    'm07': 'Analogy Mapper',
    'm08': 'Conflict Resolver',
}

def check_seed_files():
    """시드 파일 존재 여부 확인"""
    print("=" * 60)
    print("1. 시드 파일 존재 여부 확인")
    print("=" * 60)
    
    issues = []
    
    for level, seeds in EXPECTED_SEEDS.items():
        seed_dir = PROJECT_ROOT / 'seeds' / level
        print(f"\n[{level.upper()}]")
        
        for seed_id in seeds:
            seed_file = seed_dir / f"{seed_id}_{SEED_NAMES[seed_id].lower().replace(' ', '_')}.py"
            
            if seed_file.exists():
                print(f"  ✓ {seed_id.upper()}: {SEED_NAMES[seed_id]}")
            else:
                print(f"  ✗ {seed_id.upper()}: {SEED_NAMES[seed_id]} - 파일 없음")
                issues.append(f"누락된 파일: {seed_file}")
    
    return issues

def check_init_files():
    """__init__.py 파일 일관성 확인"""
    print("\n" + "=" * 60)
    print("2. __init__.py 파일 일관성 확인")
    print("=" * 60)
    
    issues = []
    
    # Molecular __init__.py 확인
    molecular_init = PROJECT_ROOT / 'seeds' / 'molecular' / '__init__.py'
    
    if molecular_init.exists():
        content = molecular_init.read_text()
        
        print("\n[Molecular __init__.py]")
        
        # 구현된 시드만 import되어야 함
        implemented = ['m01', 'm02', 'm03', 'm04']
        not_implemented = ['m05', 'm06', 'm07', 'm08']
        
        for seed_id in implemented:
            class_name = ''.join(word.capitalize() for word in SEED_NAMES[seed_id].split())
            if class_name in content:
                print(f"  ✓ {seed_id.upper()}: {class_name} import됨")
            else:
                print(f"  ✗ {seed_id.upper()}: {class_name} import 누락")
                issues.append(f"__init__.py에 {class_name} import 누락")
        
        for seed_id in not_implemented:
            class_name = ''.join(word.capitalize() for word in SEED_NAMES[seed_id].split())
            if class_name in content:
                print(f"  ⚠ {seed_id.upper()}: {class_name} - 아직 구현되지 않았는데 import됨")
                issues.append(f"__init__.py에 미구현 시드 {class_name} import됨")
    
    return issues

def check_documentation_consistency():
    """문서 일관성 확인"""
    print("\n" + "=" * 60)
    print("3. 문서 일관성 확인")
    print("=" * 60)
    
    issues = []
    
    # README.md 확인
    readme = PROJECT_ROOT / 'README.md'
    if readme.exists():
        content = readme.read_text()
        print("\n[README.md]")
        
        # 32개 시드 언급 확인
        if '32개' in content or '32 시드' in content:
            print("  ✓ 32개 시드 언급됨")
        else:
            print("  ⚠ 32개 시드 언급 누락")
        
        # Level 1 시드 테이블 확인
        for seed_id in ['m01', 'm02', 'm03', 'm04', 'm05', 'm06', 'm07', 'm08']:
            if seed_id.upper() in content or SEED_NAMES[seed_id] in content:
                print(f"  ✓ {seed_id.upper()}: {SEED_NAMES[seed_id]} 언급됨")
            else:
                print(f"  ✗ {seed_id.upper()}: {SEED_NAMES[seed_id]} 언급 누락")
                issues.append(f"README.md에 {SEED_NAMES[seed_id]} 누락")
    
    # LEVEL1_PHASE1_COMPLETE.md 확인
    phase1_doc = PROJECT_ROOT / 'LEVEL1_PHASE1_COMPLETE.md'
    if phase1_doc.exists():
        content = phase1_doc.read_text()
        print("\n[LEVEL1_PHASE1_COMPLETE.md]")
        
        # Phase 1 시드만 완료로 표시되어야 함
        implemented = ['m01', 'm02', 'm04']
        for seed_id in implemented:
            if '✓' in content and seed_id.upper() in content:
                print(f"  ✓ {seed_id.upper()}: 완료로 표시됨")
            else:
                print(f"  ⚠ {seed_id.upper()}: 완료 표시 확인 필요")
    
    return issues

def check_phase_consistency():
    """Phase 분류 일관성 확인"""
    print("\n" + "=" * 60)
    print("4. Phase 분류 일관성 확인")
    print("=" * 60)
    
    issues = []
    
    # 가이드에 따른 Phase 분류
    phases = {
        'Phase 1': ['m01', 'm02', 'm04'],
        'Phase 2': ['m03', 'm06'],
        'Phase 3': ['m05', 'm07'],
        'Phase 4': ['m08'],
    }
    
    print("\n[가이드 기준 Phase 분류]")
    for phase, seeds in phases.items():
        print(f"\n{phase}:")
        for seed_id in seeds:
            seed_file = PROJECT_ROOT / 'seeds' / 'molecular' / f"{seed_id}_{SEED_NAMES[seed_id].lower().replace(' ', '_')}.py"
            status = "✓ 구현됨" if seed_file.exists() else "✗ 미구현"
            print(f"  {status} - {seed_id.upper()}: {SEED_NAMES[seed_id]}")
    
    # Phase 1이 완료되었는지 확인
    phase1_complete = all(
        (PROJECT_ROOT / 'seeds' / 'molecular' / f"{seed_id}_{SEED_NAMES[seed_id].lower().replace(' ', '_')}.py").exists()
        for seed_id in phases['Phase 1']
    )
    
    if phase1_complete:
        print("\n✓ Phase 1 완료")
    else:
        print("\n✗ Phase 1 미완료")
        issues.append("Phase 1이 완료되지 않음")
    
    # M03이 Phase 2인 이유 확인
    print("\n[M03이 Phase 2인 이유]")
    print("  - M03은 Atomic 시드만 사용 (A03 + A06 + A01)")
    print("  - 하지만 M05 (Concept Crystallizer)가 M03에 의존")
    print("  - 따라서 M03은 Phase 2로 분류되어 Phase 3 구현 전에 완료 필요")
    
    return issues

def main():
    """메인 함수"""
    print("\n프로젝트 일관성 체크 시작...\n")
    
    all_issues = []
    
    # 1. 시드 파일 확인
    all_issues.extend(check_seed_files())
    
    # 2. __init__.py 확인
    all_issues.extend(check_init_files())
    
    # 3. 문서 일관성 확인
    all_issues.extend(check_documentation_consistency())
    
    # 4. Phase 일관성 확인
    all_issues.extend(check_phase_consistency())
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    
    if all_issues:
        print(f"\n발견된 문제: {len(all_issues)}개\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\n✓ 모든 검사 통과!")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()

