
import os
import time
import re
import json
import logging
from expert_scrutinizer import (
    _audit_linguistic_quality,
    _audit_design_and_image,
    _validate_technical_constraints,
    _manage_persona_context,
    _audit_image_placement,
    _generate_blog_tags
)
from dotenv import load_dotenv

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
REFERENCES_DIR = "references"
MAX_RETRIES = 5

# Custom Logger for 'Breathing' with the Agent
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Speech Register Definitions (6가지 어체) ---
# Agent가 콘텐츠 분석 후 자동 선택할 때 참조
SPEECH_REGISTERS = {
    1: {
        'name': '해라체',
        'level': 1,
        'description': '반말 중 가장 낮은 격식체. 권위있는 학술/전문 블로그에 적합.',
        'keywords': ['학술', '전문', '논문', '분석', '연구', '메커니즘'],
        'examples': {
            '평서': ['-다', '-ㄴ다/는다', '-았/었다', '-더라', '-구나', '-군'],
            '의문': ['-냐?', '-느냐?', '-니?', '-ㄹ까?'],
            '명령': ['-아라/어라', '-거라', '-렴'],
            '청유': ['-자']
        }
    },
    2: {
        'name': '해체',
        'level': 2,
        'description': '친근한 반말. 일상적이고 캐주얼한 블로그에 적합.',
        'keywords': ['일상', '친근', '캐주얼', '경험담', '후기'],
        'examples': {
            '평서': ['-어/아', '-지', '-거든', '-네', '-는데', '-잖아', '-더라고'],
            '의문': ['-어?/아?', '-지?', '-는데?'],
            '명령': ['-어/아'],
            '청유': ['-어/아']
        }
    },
    3: {
        'name': '하게체',
        'level': 3,
        'description': '예스러운 하대. 중년층 이상 또는 복고풍 블로그에 적합.',
        'keywords': ['복고', '고전', '역사', '전통'],
        'examples': {
            '평서': ['-네', '-겠네', '-ㄴ가/는가'],
            '의문': ['-나?', '-는가?', '-던가?'],
            '명령': ['-게', '-게나'],
            '청유': ['-세', '-세나']
        }
    },
    4: {
        'name': '하오체',
        'level': 4,
        'description': '예스러운 존대. 사극풍이나 격식있는 복고 스타일에 적합.',
        'keywords': ['사극', '역사적', '격식'],
        'examples': {
            '평서': ['-오', '-소', '-리오', '-구려'],
            '의문': ['-오?', '-소?'],
            '명령': ['-시오'],
            '청유': ['-ㅂ시다', '합시다']
        }
    },
    5: {
        'name': '해요체',
        'level': 5,
        'description': '부드러운 존댓말. 친근하면서도 예의있는 일반 블로그에 적합.',
        'keywords': ['친절', '안내', '튜토리얼', '가이드', '설명'],
        'examples': {
            '평서': ['-어요/아요', '-에요/예요', '-죠', '-지요', '-네요', '-거든요'],
            '의문': ['-어요?/아요?', '-죠?', '-나요?', '-ㄹ까요?'],
            '명령': ['-세요', '-주세요'],
            '청유': ['-어요/아요', '-ㄹ래요']
        }
    },
    6: {
        'name': '하십시오체',
        'level': 6,
        'description': '가장 격식있는 존댓말. 공식적/비즈니스 블로그에 적합.',
        'keywords': ['공식', '비즈니스', '기업', '보고서', '발표'],
        'examples': {
            '평서': ['-습니다/ㅂ니다', '-입니다', '-겠습니다'],
            '의문': ['-습니까?/ㅂ니까?', '-입니까?'],
            '명령': ['-십시오', '-소서'],
            '청유': ['-십시다']
        }
    }
}


def signal(msg):
    """Signals to the Agent in the terminal to maintain synchronization."""
    print(f"\n[AGENT-PULSE] {msg}")
    time.sleep(0.5)


def read_input(filename):
    """Read file from input directory."""
    with open(os.path.join(INPUT_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()


def read_reference(filename):
    """Read file from references directory."""
    path = os.path.join(REFERENCES_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def save_output(content, filename):
    """Save content to output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    logging.info(f"💾 Saved final HTML to {path}")


def auto_select_speech_register(content: str) -> dict:
    """
    콘텐츠 분석 후 가장 적합한 어체를 자동 선택.
    
    기본값: 해라체 (전문적/학술적 블로그 기본)
    키워드 매칭으로 다른 어체가 더 적합하면 변경.
    
    Args:
        content: 분석할 텍스트 콘텐츠
    
    Returns:
        선택된 어체 정보 dict
    """
    content_lower = content.lower()
    
    # 키워드 매칭 점수 계산
    scores = {}
    for level, info in SPEECH_REGISTERS.items():
        score = 0
        for keyword in info.get('keywords', []):
            if keyword in content_lower:
                score += 1
        scores[level] = score
    
    # 가장 높은 점수의 어체 선택 (동점이면 해라체 우선)
    best_level = 1  # 기본: 해라체
    best_score = scores.get(1, 0)
    
    for level, score in scores.items():
        if score > best_score:
            best_level = level
            best_score = score
    
    selected = SPEECH_REGISTERS[best_level]
    logging.info(f"🎯 어체 자동 선택: {selected['name']} (키워드 매칭: {best_score}개)")
    return selected


def auto_construct_persona(input_text: str) -> dict:
    """
    콘텐츠 분석 후 Writer/Reader 페르소나 자동 구성.
    
    Agent가 텍스트를 분석하여 최적의 페르소나를 자동 생성.
    사용자 입력 없이 One-Click으로 진행.
    
    Args:
        input_text: 분석할 입력 텍스트
    
    Returns:
        자동 생성된 페르소나 dict
    """
    signal("콘텐츠 분석 후 페르소나 자동 구성 중...")
    
    # 콘텐츠에서 키워드 추출하여 전문 분야 추론
    # (실제로는 LLM이 분석, 여기서는 휴리스틱 사용)
    
    # 기술 관련 키워드
    tech_keywords = ['프로그래밍', '코딩', 'SQL', '데이터베이스', 'API', '알고리즘']
    science_keywords = ['물리', '화학', '생물', '열역학', '에너지', '분자']
    business_keywords = ['마케팅', '비즈니스', '경영', '투자', '창업']
    life_keywords = ['일상', '여행', '음식', '요리', '리뷰', '후기']
    
    text_lower = input_text.lower()
    
    # 분야 판별
    expertise = "General Expert"
    if any(k in text_lower for k in tech_keywords):
        expertise = "Technology & Development"
    elif any(k in text_lower for k in science_keywords):
        expertise = "Science & Engineering"
    elif any(k in text_lower for k in business_keywords):
        expertise = "Business & Marketing"
    elif any(k in text_lower for k in life_keywords):
        expertise = "Lifestyle & Experience"
    
    # 텍스트 길이로 독자 수준 추론
    text_length = len(input_text)
    if text_length > 10000:
        reader_level = "Expert"
        mental_state = "Deep Analyst"
    elif text_length > 5000:
        reader_level = "Intermediate"
        mental_state = "Strategic Mentor"
    else:
        reader_level = "Beginner"
        mental_state = "Clear Explainer"
    
    persona = {
        "writer": {
            "expertise": expertise,
            "mental_state": mental_state,
            "tone": "",  # 어체 선택 후 설정
            "image_strategy": "Photorealistic images with handwritten Korean labels, hosted on GitHub"
        },
        "reader": {
            "background": f"Target audience for {expertise}",
            "needs": "Clear explanation with examples",
            "intellectual_level": reader_level
        }
    }
    
    logging.info(f"🎭 페르소나 자동 구성: Writer={expertise}/{mental_state}, Reader={reader_level}")
    return persona


def validate_prerequisites() -> bool:
    """
    Validate that all prerequisites are met before starting.
    
    Returns:
        True if all prerequisites are met.
    
    Raises:
        FileNotFoundError: If required files are missing.
    """
    # Check input directory
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"❌ Input directory '{INPUT_DIR}' not found.")
    
    # Check for input files
    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt') or f.endswith('.html')]
    if not input_files:
        raise FileNotFoundError(f"❌ No input files (.txt or .html) found in '{INPUT_DIR}/'")
    
    # Check references directory
    if not os.path.exists(REFERENCES_DIR):
        logging.warning(f"⚠️ References directory '{REFERENCES_DIR}' not found. Creating...")
        os.makedirs(REFERENCES_DIR)
    
    # Check output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    return True


def main():
    """
    Main orchestration function - ONE CLICK workflow.
    
    모든 설정이 자동으로 진행됨:
    1. Prerequisites 검증
    2. 콘텐츠 로드
    3. 페르소나 자동 구성 (Agent가 분석)
    4. 어체 자동 선택 (Agent가 분석)
    5. Scrutiny Loop 실행
    6. 결과 저장
    """
    signal("Entering One-Click Meta Pipeline: 'BLOG FACTORY' v2.0")
    
    # Step 0: Validate prerequisites
    try:
        validate_prerequisites()
    except FileNotFoundError as e:
        logging.error(str(e))
        return
    
    # Load input content
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    html_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.html')]
    
    original_text = ""
    if files:
        original_text = read_input(files[0])
    
    # Step 1: AUTO - Construct Persona
    signal("Step 1: 페르소나 자동 구성 (Agent Analyzing...)")
    persona = auto_construct_persona(original_text)
    
    # Step 2: AUTO - Select Speech Register
    signal("Step 2: 어체 자동 선택 (Agent Analyzing...)")
    speech_register = auto_select_speech_register(original_text)
    
    # Sync persona with speech register
    persona['writer']['tone'] = f"{speech_register['name']} 사용"
    persona['speech_register'] = speech_register
    
    _manage_persona_context(persona)
    signal("페르소나 및 어체 자동 설정 완료. Engine Synchronized.")
    
    # Display auto-configuration summary
    print("\n" + "="*60)
    print("✅ 자동 설정 완료 (Auto-Configuration Complete)")
    print("="*60)
    print(f"  📝 어체: {speech_register['name']}")
    print(f"  🎭 Writer: {persona['writer']['expertise']} / {persona['writer']['mental_state']}")
    print(f"  👤 Reader: {persona['reader']['intellectual_level']}")
    print(f"  📌 허용 종결어미: {', '.join(speech_register['examples'].get('평서', [])[:4])}")
    print("="*60 + "\n")
    
    # Step 3: Load HTML content
    if html_files:
        with open(os.path.join(INPUT_DIR, html_files[0]), 'r', encoding='utf-8') as f:
            current_html = f.read()
        filename = html_files[0]
    else:
        signal("No HTML file found. HTML generation from TXT is required.")
        filename = files[0].replace('.txt', '.html') if files else "output.html"
        current_html = "<!-- Generated Draft Placeholder -->"
    
    # Step 4: Recursive Scrutiny Loop
    signal("Beginning Scrutiny Loop. Standing by for reports.")
    for attempt in range(1, MAX_RETRIES + 1):
        logging.info(f"🔄 Scrutiny Cycle #{attempt} Start")
        
        # Engineering + Linguistic + Image Scrutiny
        report = "\n".join([
            _audit_linguistic_quality(current_html, persona),
            _audit_design_and_image(current_html, json.dumps(persona)),
            _audit_image_placement(current_html, persona),
            _validate_technical_constraints(current_html)
        ])
        
        print("\n" + "-"*40)
        print(f"📋 Scrutiny Report (Cycle #{attempt})")
        print("-"*40)
        print(report)
        print("-"*40 + "\n")
        
        if "❌" in report:
            signal(f"Red Signal (❌) on Cycle #{attempt}. Manual/LLM correction required...")
            # In auto mode, this would call LLM for correction
            break
        else:
            signal(f"Green Signal (✅) on Cycle #{attempt}. Content logic verified.")
            break
    
    # Step 4.5: Generate Blog Tags (Auto Briefing)
    signal("태그 자동 생성 및 브리핑 중...")
    tags_briefing = _generate_blog_tags(current_html, persona)
    print("\n" + "="*60)
    print(tags_briefing)
    print("="*60 + "\n")
    
    # Step 5: Final Save
    save_output(current_html, f"final_{filename}")
    signal("Process Complete. Check 'output/' directory for results.")


if __name__ == "__main__":
    main()
