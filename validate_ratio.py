#!/usr/bin/env python3
"""
절:구 비율 검증 모듈

워크플로우:
1. Agent가 input/ 폴더의 콘텐츠를 읽음
2. 시스템이 절 수를 자동 계산하여 Agent에게 보고
3. Agent가 구(Phrase) 수를 직접 세어 입력
4. 시스템이 절:구 비율 검증 후 결과 반환

목표 비율: 절:구 = 2:1 (±0.3 오차 허용, 즉 1.7~2.3 범위)

사용법:
  - CLI: python validate_ratio.py <html_file> [phrase_count]
  - 모듈: from validate_ratio import count_clauses, validate_ratio
"""

import sys
import re
import os
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Jamo Decomposition (expert_scrutinizer.py와 동일)
# -----------------------------------------------------------------------------
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def decompose_hangul(text: str) -> str:
    """한글을 자모로 분해."""
    result = []
    for char in text:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            cho = offset // (21 * 28)
            jung = (offset % (21 * 28)) // 28
            jong = offset % 28
            result.append(CHOSUNG[cho])
            result.append(JUNGSUNG[jung])
            if jong > 0:
                result.append(JONGSUNG[jong])
        else:
            result.append(char)
    return ''.join(result)


# -----------------------------------------------------------------------------
# 절(Clause) 수 계산 - 시스템 자동
# -----------------------------------------------------------------------------
def count_clauses(content: str) -> dict:
    """
    HTML 또는 텍스트에서 절 수를 자동 계산.
    
    Args:
        content: HTML 콘텐츠 또는 순수 텍스트
        
    Returns:
        dict: {
            'terminal': 종결어미(독립절) 수,
            'embedded': 관형절(안은문장) 수,
            'total': 총 절 수
        }
    """
    # HTML인 경우 텍스트 추출
    if '<' in content and '>' in content:
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
    else:
        text = content
    
    jamo = decompose_hangul(text)
    
    # 종결어미 패턴 (독립절)
    terminal_patterns = [
        r'ㄷㅏ[\.!]', r'ㅇㅓ[\.!]', r'ㅇㅏ[\.!]', r'ㅈㅣ[\.!]',
        r'ㄴㅔ[\.!]', r'ㅇㅛ[\.!]', r'ㄴㅣㄷㅏ[\.!]',
        r'ㄴㅑ[\?]', r'ㄴㅣ[\?]', r'ㅈㅏ[\.!]', r'ㄹㅏ[\.!]',
    ]
    terminal = sum(len(re.findall(p, jamo)) for p in terminal_patterns)
    
    # 관형절 (안은문장)
    embedded = len(re.findall(
        r'(?:는|ㄴ|은|ㄹ|을|던)\s*(?:것|거|때|곳|이|수|줄|법|리)', 
        text
    ))
    
    return {
        'terminal': terminal,
        'embedded': embedded,
        'total': terminal + embedded
    }


def count_clauses_from_file(file_path: str) -> dict:
    """파일에서 절 수 계산."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return count_clauses(content)


# -----------------------------------------------------------------------------
# 절:구 비율 검증
# -----------------------------------------------------------------------------
TARGET_RATIO = 2.0  # 목표 비율 절:구 = 2:1
TOLERANCE = 0.3     # 오차 허용 범위 (1.7 ~ 2.3)


def validate_ratio(clause_count: int, phrase_count: int) -> dict:
    """
    절:구 비율 검증.
    
    Agent 워크플로우:
    1. 시스템이 clause_count를 계산하여 제공
    2. Agent가 phrase_count를 직접 세어 입력
    3. 이 함수가 비율 검증 후 결과 반환
    
    Args:
        clause_count: 시스템이 계산한 절 수
        phrase_count: Agent가 입력한 구 수
        
    Returns:
        dict: {
            'pass': bool,
            'ratio': float,
            'clause_count': int,
            'phrase_count': int,
            'message': str,
            'action': str (실패시 수정 지시)
        }
    """
    if phrase_count <= 0:
        return {
            'pass': False,
            'ratio': float('inf'),
            'clause_count': clause_count,
            'phrase_count': phrase_count,
            'message': '❌ ERROR: 구 수가 0 이하입니다.',
            'action': '구(명사구/동사구/부사구)를 다시 계산하세요.'
        }
    
    ratio = clause_count / phrase_count
    min_ratio = TARGET_RATIO - TOLERANCE  # 1.7
    max_ratio = TARGET_RATIO + TOLERANCE  # 2.3
    
    if min_ratio <= ratio <= max_ratio:
        return {
            'pass': True,
            'ratio': ratio,
            'clause_count': clause_count,
            'phrase_count': phrase_count,
            'message': f'✅ PASS: 절:구 = {ratio:.2f}:1 (허용 범위: {min_ratio}~{max_ratio})',
            'action': None
        }
    else:
        if ratio > max_ratio:
            # 구가 부족
            ideal_phrases = int(clause_count / TARGET_RATIO)
            needed = ideal_phrases - phrase_count
            action = f'구 {needed}개 추가 필요. 복합 명사구/부사구 사용 권장.'
        else:
            # 구가 과다
            ideal_phrases = int(clause_count / TARGET_RATIO)
            excess = phrase_count - ideal_phrases
            action = f'구 {excess}개 축소 또는 절로 확장 필요.'
        
        return {
            'pass': False,
            'ratio': ratio,
            'clause_count': clause_count,
            'phrase_count': phrase_count,
            'message': f'❌ FAIL: 절:구 = {ratio:.2f}:1 (허용 범위: {min_ratio}~{max_ratio})',
            'action': action
        }


# -----------------------------------------------------------------------------
# Agent용 보고 형식 생성
# -----------------------------------------------------------------------------
def generate_clause_report(file_path: str) -> str:
    """
    Agent에게 전달할 절 수 보고서 생성.
    
    Agent는 이 보고서를 받은 후 직접 구를 세어 validate_ratio()에 입력해야 함.
    """
    clauses = count_clauses_from_file(file_path)
    
    report = []
    report.append("=" * 50)
    report.append("📊 [시스템] 절 수 자동 계산 완료")
    report.append("=" * 50)
    report.append(f"📄 파일: {file_path}")
    report.append(f"ℹ️  종결어미(독립절): {clauses['terminal']}개")
    report.append(f"ℹ️  관형절(안은문장): {clauses['embedded']}개")
    report.append(f"ℹ️  총 절 수: {clauses['total']}개")
    report.append("-" * 50)
    report.append("")
    report.append("📋 [Agent Task] 이제 구(Phrase)를 직접 세고 아래 형식으로 보고:")
    report.append("   (구 = 술어 기능 없는 단위)")
    report.append("   → 명사구(NP): [발견한 구 나열] → ___개")
    report.append("   → 동사구(VP): [발견한 구 나열] → ___개")
    report.append("   → 부사구(AP): [발견한 구 나열] → ___개")
    report.append("   → 구 총계: ___개")
    report.append("")
    report.append(f"📝 목표 비율: 절:구 = 2:1 (허용 범위: 1.7~2.3)")
    report.append(f"📝 현재 절 수 기준 이상적인 구 수: {int(clauses['total'] / TARGET_RATIO)}개 ±{int(clauses['total'] * TOLERANCE / TARGET_RATIO)}개")
    report.append("=" * 50)
    
    return "\n".join(report)


# -----------------------------------------------------------------------------
# CLI 인터페이스
# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_ratio.py <html_file> [phrase_count]")
        print("")
        print("Examples:")
        print("  python validate_ratio.py output/final_sample.html")
        print("  python validate_ratio.py output/final_sample.html 45")
        print("")
        print("Mode 1 (no phrase_count): 절 수만 계산하고 Agent에게 보고")
        print("Mode 2 (with phrase_count): 절:구 비율 검증 실행")
        sys.exit(1)
    
    html_path = sys.argv[1]
    
    if not os.path.exists(html_path):
        print(f"❌ ERROR: 파일을 찾을 수 없습니다: {html_path}")
        sys.exit(1)
    
    # 절 수 계산
    clauses = count_clauses_from_file(html_path)
    
    if len(sys.argv) < 3:
        # Mode 1: 절 수만 보고 (Agent가 구 수를 세기 위한 정보 제공)
        print(generate_clause_report(html_path))
        sys.exit(0)
    else:
        # Mode 2: 비율 검증
        try:
            phrase_count = int(sys.argv[2])
        except ValueError:
            print(f"❌ ERROR: 구 수는 정수여야 합니다: {sys.argv[2]}")
            sys.exit(1)
        
        print("=" * 50)
        print("📊 절:구 비율 검증")
        print("=" * 50)
        print(f"📄 파일: {html_path}")
        print(f"ℹ️  시스템 계산 절 수: {clauses['total']}개")
        print(f"📥 Agent 입력 구 수: {phrase_count}개")
        print("-" * 50)
        
        result = validate_ratio(clauses['total'], phrase_count)
        print(result['message'])
        if result['action']:
            print(f"📝 수정 지시: {result['action']}")
        print("=" * 50)
        
        sys.exit(0 if result['pass'] else 1)


if __name__ == "__main__":
    main()
