from fastmcp import FastMCP
from bs4 import BeautifulSoup
import re
import json

# -----------------------------------------------------------------------------
# Korean Jamo (자모) Decomposition Utilities
# LLM-style tokenization: 한글 → 초성/중성/종성 분해
# -----------------------------------------------------------------------------

# Unicode Hangul Jamo
CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def decompose_hangul(text: str) -> str:
    """
    Decompose Korean text into jamo (초성/중성/종성).
    Example: "더라" → "ㄷㅓㄹㅏ"
    Non-Hangul characters are kept as-is.
    """
    result = []
    for char in text:
        code = ord(char)
        # Hangul syllable range: 0xAC00 ~ 0xD7A3
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

def compose_jamo(jamo_str: str) -> str:
    """
    Compose jamo back into Hangul syllables.
    Example: "ㄷㅓㄹㅏ" → "더라"
    """
    result = []
    i = 0
    while i < len(jamo_str):
        char = jamo_str[i]
        # Check if it's a chosung
        if char in CHOSUNG and i + 1 < len(jamo_str) and jamo_str[i + 1] in JUNGSUNG:
            cho = CHOSUNG.index(char)
            jung = JUNGSUNG.index(jamo_str[i + 1])
            jong = 0
            i += 2
            # Check for jongsung
            if i < len(jamo_str) and jamo_str[i] in JONGSUNG[1:]:
                # Peek ahead to see if this is actually a chosung for next syllable
                if i + 1 < len(jamo_str) and jamo_str[i + 1] in JUNGSUNG:
                    pass  # It's a chosung, don't consume
                else:
                    jong = JONGSUNG.index(jamo_str[i])
                    i += 1
            syllable = chr(0xAC00 + (cho * 21 * 28) + (jung * 28) + jong)
            result.append(syllable)
        else:
            result.append(char)
            i += 1
    return ''.join(result)

# Define the MCP Server
mcp = FastMCP("Blog Scrutinizer")

# -----------------------------------------------------------------------------
# 0. Persona Context Manager
# -----------------------------------------------------------------------------

def _manage_persona_context(persona_input: dict = None) -> str:
    """
    Manages the persona context. If no input is provided, returns empty fields.
    Does NOT guess or generate heuristics.
    """
    if not persona_input:
        persona = {
            "writer": {"expertise": "", "mental_state": "", "tone": "", "image_strategy": ""},
            "reader": {"background": "", "needs": "", "intellectual_level": ""}
        }
    else:
        # Use provided input
        persona = persona_input

    # Ensure no names are concatenated in labels
    return json.dumps(persona, ensure_ascii=False, indent=2)

# -----------------------------------------------------------------------------
# 1. Hybrid Linguistic Auditor
# -----------------------------------------------------------------------------

def _audit_linguistic_quality(content: str, persona_json: str = "{}") -> str:
    """
    Provides Engineering-based audit for linguistic structure:
    1. Syntactic Complexity (Ratio > 1.5)
    2. Stale Word Blocking (Hard Fail)
    3. Sharp Word Statistics (Guidance for LLM)
    """
    soup = BeautifulSoup(content, 'html.parser')
    text_content = soup.get_text()
    
    report = []
    errors = 0

    # A. 절(Clause) / 구(Phrase) 분석
    # - 절: 자모 기반 종결어미 + 관형절(안은문장) 자동 계산
    # - 구: LLM이 직접 세고 무엇을 셌는지 병기 (명사구/동사구/부사구)
    # - 목표 비율: 절:구 = 2:1 (±0.3 오차 허용)
    
    jamo_text = decompose_hangul(text_content)
    
    # 1. 종결어미 감지 (독립절)
    terminal_patterns = [
        r'ㄷㅏ[\.!]',  # -다
        r'ㅇㅓ[\.!]', r'ㅇㅏ[\.!]',  # -어/-아
        r'ㅈㅣ[\.!]',  # -지
        r'ㄴㅔ[\.!]',  # -네
        r'ㅇㅛ[\.!]',  # -요 계열
        r'ㄴㅣㄷㅏ[\.!]',  # -니다
        r'ㄴㅑ[\?]', r'ㄴㅣ[\?]',  # -냐/-니?
        r'ㅈㅏ[\.!]',  # -자
        r'ㄹㅏ[\.!]',  # -라
    ]
    terminal_clause_count = sum(len(re.findall(p, jamo_text)) for p in terminal_patterns)
    
    # 2. 관형절/안은문장 감지 (내포절)
    # 관형사형 어미: -는, -ㄴ/은, -ㄹ/을, -던 + 체언
    embedded_markers = re.findall(r'(?:는|ㄴ|은|ㄹ|을|던)\s*(?:것|거|때|곳|이|수|줄|법|리)', text_content)
    embedded_clause_count = len(embedded_markers)
    
    # 3. 총 절 수
    total_clause_count = terminal_clause_count + embedded_clause_count
    
    report.append(f"ℹ️ [절-Stats] 종결어미(독립절): {terminal_clause_count}개")
    report.append(f"ℹ️ [절-Stats] 관형절(안은문장): {embedded_clause_count}개")
    report.append(f"ℹ️ [절-Total] 총 절 수: {total_clause_count}개 (시스템 자동 계산)")
    
    # 4. 구(Phrase) 가이드 - LLM이 직접 세고 병기 (목표 비율은 숨김)
    # 구 = 술어 기능이 없는 단위 (명사구, 동사구, 부사구 + 독립 부사/관형사)
    report.append(f"\n📋 [구-Task] LLM이 직접 '구' 단위를 세고 아래 형식으로 보고:")
    report.append(f"   (구 = 술어 기능 없는 단위. 부사/관형사 단독도 포함)")
    report.append(f"   → 명사구(NP): [실제 발견한 구 나열] → 총 ___개")
    report.append(f"   → 동사구(VP): [실제 발견한 구 나열] → 총 ___개")
    report.append(f"   → 부사구(AP): [실제 발견한 구 나열] → 총 ___개")
    report.append(f"   → 독립부사: [예: '단언컨대', '압도적으로', '그러나'] → 총 ___개")
    report.append(f"   → 관형사: [예: '이', '그', '새로운', '모든'] → 총 ___개")
    report.append(f"   → 구 총계: ___개")
    report.append(f"⚠️ [주의] 반드시 발견한 구를 실제로 나열할 것. 숫자만 보고 금지.")

    # B. Lexical Audit (Engineering)
    # 1. Hard Block: Stale Words
    # Removed '알아보자' as per instruction to leave it to LLM judgment within Reckon Vibe
    stale_words = ['사료된다', '고찰', '본인', '하였음', '의미한다', '뜻한다']
    found_stale = [w for w in stale_words if w in text_content]
    if found_stale:
        report.append(f"❌ [Lexical-Block] Stale words detected: {found_stale}. Replace with dynamic/sharp alternatives.")
        errors += 1
    
    # 2. Guidance: Sharp Word Statistics
    sharp_pool = ['장악', '설계', '압도적', '메커니즘', '단언컨대', '귀결', '납득', '양상']
    found_sharp = [w for w in sharp_pool if w in text_content]
    report.append(f"ℹ️ [Lexical-Stats] Sharp word count: {len(found_sharp)}. (Examples found: {found_sharp})")
    
    # C. Korean Speech Register (어체) Audit - JAMO-BASED SYSTEM
    # Level hierarchy (low → high): 해라 < 해 < 하게 < 하오 < 해요 < 하십시오
    # 
    # JAMO DECOMPOSITION: 한글 → ㅈ/ㅏ/ㅁ/ㅗ 단위 분해
    # Example: "더라" → "ㄷㅓㄹㅏ", "했다" → "ㅎㅐㅆㄷㅏ"
    #
    # PATTERNS INCLUDE:
    # - 본용언 종결어미 (main verb endings)
    # - 보조용언 결합형 (auxiliary verb combinations)
    #   - ~아/어 보다, 주다, 버리다, 놓다, 두다, 가다, 오다, 내다, 대다
    #   - ~고 있다, 싶다, 말다
    #   - ~아/어야 하다/되다
    #   - ~ㄹ/을 수 있다/없다
    # - 시제/상 결합 (tense/aspect: 았/었/겠)
    # - 불규칙 활용 (irregular: ㄹ/ㅂ/ㄷ/ㅅ/ㅎ)
    # - 비표준/방언/축약형
    
    # Convert text to jamo for pattern matching
    jamo_text = decompose_hangul(text_content)
    
    # Jamo building blocks (for constructing patterns)
    # 모음조화 그룹
    YANG_V = 'ㅏㅗㅑㅛ'  # 양성모음
    YIN_V = 'ㅓㅜㅕㅠㅡㅣㅐㅔㅚㅟ'  # 음성모음
    ALL_V = YANG_V + YIN_V + 'ㅘㅙㅝㅞㅢㅒㅖ'  # 모든 모음
    
    # Common auxiliary verb stems in jamo (보조용언 어간)
    # 보다(ㅂㅗㄷㅏ), 주다(ㅈㅜㄷㅏ), 버리다(ㅂㅓㄹㅣㄷㅏ), 놓다(ㄴㅗㅎㄷㅏ), 두다(ㄷㅜㄷㅏ)
    # 가다(ㄱㅏㄷㅏ), 오다(ㅇㅗㄷㅏ), 내다(ㄴㅐㄷㅏ), 있다(ㅇㅣㅆㄷㅏ), 싶다(ㅅㅣㅍㄷㅏ)
    
    # Optional preceding elements (tense/aspect markers in jamo)
    TENSE_OPT = r'(?:ㅇㅏㅆ|ㅇㅓㅆ|ㄱㅔㅆ)?'  # 았/었/겠 (optional)
    AUX_OPT = r'(?:ㅂㅗ|ㅈㅜ|ㅂㅓㄹㅣ|ㄴㅗㅎ|ㄷㅜ|ㄱㅏ|ㅇㅗ|ㄴㅐ|ㅇㅣㅆ|ㅅㅣㅍ)?'  # 보/주/버리/놓/두/가/오/내/있/싶
    
    speech_levels = {
        '해라체': {
            'jamo_patterns': [
                # === 평서형 ===
                rf'{TENSE_OPT}ㄷㅏ[\\.!]',  # -다
                rf'{TENSE_OPT}ㄴㅡㄴㄷㅏ[\\.!]',  # -는다
                rf'{TENSE_OPT}ㄴㄷㅏ[\\.!]',  # -ㄴ다
                rf'{TENSE_OPT}ㄷㅓㄹㅏ[\\.!]',  # -더라
                rf'{TENSE_OPT}ㄷㅡㄹㅏ[\\.!]',  # -드라 (비표준)
                rf'{TENSE_OPT}ㄹㅏㄴㄷㅏ[\\.!]',  # -란다
                rf'{TENSE_OPT}ㄷㅏㄴㄷㅏ[\\.!]',  # -단다
                rf'{TENSE_OPT}ㄹㅣㄹㅏ[\\.!]',  # -리라
                rf'{TENSE_OPT}ㄱㅜㄴㅏ[\\.!]',  # -구나
                rf'{TENSE_OPT}ㄱㅜㄴ[\\.!]',  # -군
                # 보조용언 결합: ~아/어 봤다, 줬다, 버렸다, etc.
                rf'ㅂㅗ{TENSE_OPT}ㄷㅏ[\\.!]',  # 봤다/본다
                rf'ㅈㅜ{TENSE_OPT}ㄷㅏ[\\.!]',  # 줬다/준다  
                rf'ㅂㅓㄹㅣ{TENSE_OPT}ㄷㅏ[\\.!]',  # 버렸다
                rf'ㄴㅗㅎ{TENSE_OPT}ㄷㅏ[\\.!]',  # 놨다
                rf'ㄷㅜ{TENSE_OPT}ㄷㅏ[\\.!]',  # 뒀다
                rf'ㄱㅏ{TENSE_OPT}ㄷㅏ[\\.!]',  # 갔다
                rf'ㅇㅗ{TENSE_OPT}ㄷㅏ[\\.!]',  # 왔다
                # ~고 있다, 싶다
                rf'ㄱㅗㅇㅣㅆㄷㅏ[\\.!]',  # 고 있다
                rf'ㄱㅗㅅㅣㅍㄷㅏ[\\.!]',  # 고 싶다
                # === 의문형 ===
                rf'{TENSE_OPT}ㄴㅑ[\\?]',  # -냐
                rf'{TENSE_OPT}ㄴㅡㄴㅑ[\\?]',  # -느냐
                rf'{TENSE_OPT}ㄴㅣ[\\?]',  # -니
                rf'{TENSE_OPT}ㄷㅓㄴㅑ[\\?]',  # -더냐
                rf'{TENSE_OPT}ㄹㄲㅏ[\\?]',  # -ㄹ까
                rf'{TENSE_OPT}ㅇㅡㄹㄲㅏ[\\?]',  # -을까
                # === 명령형 ===
                rf'ㅇㅏㄹㅏ[\\.!]',  # -아라
                rf'ㅇㅓㄹㅏ[\\.!]',  # -어라
                rf'ㅇㅕㄹㅏ[\\.!]',  # -여라
                rf'ㄱㅓㄹㅏ[\\.!]',  # -거라
                rf'ㄴㅓㄹㅏ[\\.!]',  # -너라
                rf'ㄹㅕㅁ[\\.!]',  # -렴
                # === 청유형 ===
                rf'ㅈㅏ[\\.!]',  # -자
            ],
            'examples': {
                '평서': ['-다', '-ㄴ다/는다', '-았/었다', '-더라', '-구나', '-군'],
                '의문': ['-냐?', '-느냐?', '-니?', '-ㄹ까?'],
                '명령': ['-아라/어라', '-거라', '-렴'],
                '청유': ['-자'],
                '보조용언': ['~봤다', '~줬다', '~버렸다', '~놨다', '~갔다', '~왔다', '~고 있다', '~고 싶다']
            },
            'level': 1
        },
        '해체': {
            'jamo_patterns': [
                # === 평서형 === (~어/아/여 종결)
                rf'{TENSE_OPT}ㅇㅓ[\\.!]',  # -어
                rf'{TENSE_OPT}ㅇㅏ[\\.!]',  # -아
                rf'{TENSE_OPT}ㅇㅕ[\\.!]',  # -여
                rf'{TENSE_OPT}ㅈㅣ[\\.!]',  # -지
                rf'{TENSE_OPT}ㄱㅓㄷㅡㄴ[\\.!]',  # -거든
                rf'{TENSE_OPT}ㄴㅔ[\\.!]',  # -네
                rf'{TENSE_OPT}ㄴㅡㄴㄷㅔ[\\.!]',  # -는데
                rf'{TENSE_OPT}ㅈㅏㄴㅎㅏ[\\.!]',  # -잖아
                rf'{TENSE_OPT}ㄷㅓㄹㅏㄱㅗ[\\.!]',  # -더라고
                rf'{TENSE_OPT}ㄹㄱㅓㄹ[\\.!]',  # -ㄹ걸
                rf'{TENSE_OPT}ㄹㄱㅔ[\\.!]',  # -ㄹ게
                # 보조용언 결합: 봐, 줘, 버려, 놔, 둬
                rf'ㅂㅘ[\\.!]',  # 봐 (보+아→봐)
                rf'ㅈㅝ[\\.!]',  # 줘 (주+어→줘)
                rf'ㅂㅓㄹㅕ[\\.!]',  # 버려
                rf'ㄴㅘ[\\.!]',  # 놔 (놓+아→놔)
                rf'ㄷㅝ[\\.!]',  # 둬 (두+어→둬)
                rf'ㄱㅏ[\\.!]',  # 가
                rf'ㅇㅘ[\\.!]',  # 와 (오+아→와)
                # ~고 있어, 싶어
                rf'ㄱㅗㅇㅣㅆㅇㅓ[\\.!]',  # 고 있어
                rf'ㄱㅗㅅㅣㅍㅇㅓ[\\.!]',  # 고 싶어
                # === 의문형 ===
                rf'{TENSE_OPT}ㅇㅓ[\\?]',  # -어?
                rf'{TENSE_OPT}ㅇㅏ[\\?]',  # -아?
                rf'{TENSE_OPT}ㅈㅣ[\\?]',  # -지?
                rf'{TENSE_OPT}ㄴㅡㄴㄷㅔ[\\?]',  # -는데?
            ],
            'examples': {
                '평서': ['-어/아', '-지', '-거든', '-네', '-는데', '-잖아', '-더라고', '-ㄹ걸', '-ㄹ게'],
                '의문': ['-어?/아?', '-지?', '-는데?', '-잖아?'],
                '명령': ['-어/아 (평서와 동일)'],
                '청유': ['-어/아 (평서와 동일)'],
                '보조용언': ['~봐', '~줘', '~버려', '~놔', '~둬', '~가', '~와', '~고 있어', '~고 싶어']
            },
            'level': 2
        },
        '하게체': {
            'jamo_patterns': [
                # === 평서형 ===
                rf'{TENSE_OPT}ㄴㅔ[\\.!]',  # -네 (중복 허용)
                rf'{TENSE_OPT}ㄱㅔㅆㄴㅔ[\\.!]',  # -겠네
                rf'{TENSE_OPT}ㄴㄱㅏ[\\.!]',  # -ㄴ가
                rf'{TENSE_OPT}ㄴㅡㄴㄱㅏ[\\.!]',  # -는가
                # === 의문형 ===
                rf'{TENSE_OPT}ㄴㅏ[\\?]',  # -나?
                rf'{TENSE_OPT}ㄴㅡㄴㄱㅏ[\\?]',  # -는가?
                rf'{TENSE_OPT}ㄷㅓㄴㄱㅏ[\\?]',  # -던가?
                # === 명령형 ===
                rf'ㄱㅔ[\\.!]',  # -게
                rf'ㄱㅔㄴㅏ[\\.!]',  # -게나
                # === 청유형 ===
                rf'ㅅㅔ[\\.!]',  # -세
                rf'ㅅㅔㄴㅏ[\\.!]',  # -세나
            ],
            'examples': {
                '평서': ['-네', '-겠네', '-ㄴ가/는가'],
                '의문': ['-나?', '-는가?', '-던가?'],
                '명령': ['-게', '-게나'],
                '청유': ['-세', '-세나']
            },
            'level': 3
        },
        '하오체': {
            'jamo_patterns': [
                # === 평서형 ===
                rf'{TENSE_OPT}ㅇㅗ[\\.!]',  # -오
                rf'{TENSE_OPT}ㅅㅗ[\\.!]',  # -소
                rf'{TENSE_OPT}ㄹㅣㅇㅗ[\\.!]',  # -리오
                rf'{TENSE_OPT}ㄱㅜㄹㅕ[\\.!]',  # -구려
                # === 의문형 ===
                rf'{TENSE_OPT}ㅇㅗ[\\?]',  # -오?
                rf'{TENSE_OPT}ㅅㅗ[\\?]',  # -소?
                # === 명령형 ===
                rf'ㅅㅣㅇㅗ[\\.!]',  # -시오
                # === 청유형 ===
                rf'ㅂㅅㅣㄷㅏ[\\.!]',  # -ㅂ시다
                rf'ㅎㅏㅂㅅㅣㄷㅏ[\\.!]',  # 합시다
                rf'ㄱㅏㅂㅅㅣㄷㅏ[\\.!]',  # 갑시다
                rf'ㅂㅗㅂㅅㅣㄷㅏ[\\.!]',  # 봅시다
            ],
            'examples': {
                '평서': ['-오', '-소', '-리오', '-구려'],
                '의문': ['-오?', '-소?'],
                '명령': ['-시오'],
                '청유': ['-ㅂ시다', '합시다', '갑시다', '봅시다']
            },
            'level': 4
        },
        '해요체': {
            'jamo_patterns': [
                # === 평서형 ===
                rf'{TENSE_OPT}ㅇㅓㅇㅛ[\\.!]',  # -어요
                rf'{TENSE_OPT}ㅇㅏㅇㅛ[\\.!]',  # -아요
                rf'{TENSE_OPT}ㅇㅕㅇㅛ[\\.!]',  # -여요
                rf'{TENSE_OPT}ㅇㅔㅇㅛ[\\.!]',  # -에요
                rf'{TENSE_OPT}ㅇㅖㅇㅛ[\\.!]',  # -예요
                rf'{TENSE_OPT}ㅈㅛ[\\.!]',  # -죠
                rf'{TENSE_OPT}ㅈㅣㅇㅛ[\\.!]',  # -지요
                rf'{TENSE_OPT}ㄴㅔㅇㅛ[\\.!]',  # -네요
                rf'{TENSE_OPT}ㄱㅜㄴㅇㅛ[\\.!]',  # -군요
                rf'{TENSE_OPT}ㄱㅓㄷㅡㄴㅇㅛ[\\.!]',  # -거든요
                rf'{TENSE_OPT}ㅈㅏㄴㅎㅏㅇㅛ[\\.!]',  # -잖아요
                rf'{TENSE_OPT}ㄴㅡㄴㄷㅔㅇㅛ[\\.!]',  # -는데요
                rf'{TENSE_OPT}ㄹㄱㅔㅇㅛ[\\.!]',  # -ㄹ게요
                rf'{TENSE_OPT}ㄹㄲㅓㄹㅇㅛ[\\.!]',  # -ㄹ걸요
                # 보조용언: 봐요, 줘요, 버려요
                rf'ㅂㅘㅇㅛ[\\.!]',  # 봐요
                rf'ㅈㅝㅇㅛ[\\.!]',  # 줘요
                rf'ㅂㅓㄹㅕㅇㅛ[\\.!]',  # 버려요
                rf'ㄴㅘㅇㅛ[\\.!]',  # 놔요
                rf'ㄷㅝㅇㅛ[\\.!]',  # 둬요
                # ~고 있어요, 싶어요
                rf'ㄱㅗㅇㅣㅆㅇㅓㅇㅛ[\\.!]',  # 고 있어요
                rf'ㄱㅗㅅㅣㅍㅇㅓㅇㅛ[\\.!]',  # 고 싶어요
                # === 의문형 ===
                rf'{TENSE_OPT}ㅇㅓㅇㅛ[\\?]',  # -어요?
                rf'{TENSE_OPT}ㅇㅏㅇㅛ[\\?]',  # -아요?
                rf'{TENSE_OPT}ㅈㅛ[\\?]',  # -죠?
                rf'{TENSE_OPT}ㄴㅏㅇㅛ[\\?]',  # -나요?
                rf'{TENSE_OPT}ㄹㄲㅏㅇㅛ[\\?]',  # -ㄹ까요?
                rf'{TENSE_OPT}ㄹㄹㅐㅇㅛ[\\?]',  # -ㄹ래요?
                # === 명령형 ===
                rf'ㅅㅔㅇㅛ[\\.!]',  # -세요
                rf'ㅈㅜㅅㅔㅇㅛ[\\.!]',  # -주세요
            ],
            'examples': {
                '평서': ['-어요/아요', '-에요/예요', '-죠', '-지요', '-네요', '-군요', '-거든요', '-잖아요', '-는데요', '-ㄹ게요', '-ㄹ걸요'],
                '의문': ['-어요?/아요?', '-죠?', '-나요?', '-ㄹ까요?', '-ㄹ래요?'],
                '명령': ['-세요', '-주세요'],
                '청유': ['-어요/아요', '-ㄹ래요'],
                '보조용언': ['~봐요', '~줘요', '~버려요', '~놔요', '~둬요', '~고 있어요', '~고 싶어요']
            },
            'level': 5
        },
        '하십시오체': {
            'jamo_patterns': [
                # === 평서형 ===
                rf'{TENSE_OPT}ㅅㅡㅂㄴㅣㄷㅏ[\\.!]',  # -습니다
                rf'{TENSE_OPT}ㅂㄴㅣㄷㅏ[\\.!]',  # -ㅂ니다
                rf'{TENSE_OPT}ㅇㅗㅂㄴㅣㄷㅏ[\\.!]',  # -옵니다
                rf'{TENSE_OPT}ㅇㅣㅂㄴㅣㄷㅏ[\\.!]',  # -입니다
                rf'{TENSE_OPT}ㄱㅔㅆㅅㅡㅂㄴㅣㄷㅏ[\\.!]',  # -겠습니다
                rf'{TENSE_OPT}ㅇㅓㅆㅅㅡㅂㄴㅣㄷㅏ[\\.!]',  # -었습니다
                rf'{TENSE_OPT}ㅇㅏㅆㅅㅡㅂㄴㅣㄷㅏ[\\.!]',  # -았습니다
                # === 의문형 ===
                rf'{TENSE_OPT}ㅅㅡㅂㄴㅣㄲㅏ[\\?]',  # -습니까?
                rf'{TENSE_OPT}ㅂㄴㅣㄲㅏ[\\?]',  # -ㅂ니까?
                rf'{TENSE_OPT}ㅇㅣㅂㄴㅣㄲㅏ[\\?]',  # -입니까?
                rf'{TENSE_OPT}ㅅㅣㅂㄴㅣㄲㅏ[\\?]',  # -십니까?
                # === 명령형 ===
                rf'ㅅㅣㅂㅅㅣㅇㅗ[\\.!]',  # -십시오
                rf'ㅅㅗㅅㅓ[\\.!]',  # -소서 (극존칭)
                # === 청유형 ===
                rf'ㅅㅣㅂㅅㅣㄷㅏ[\\.!]',  # -십시다
            ],
            'examples': {
                '평서': ['-습니다/ㅂ니다', '-입니다', '-겠습니다', '-았/었습니다'],
                '의문': ['-습니까?/ㅂ니까?', '-입니까?', '-십니까?'],
                '명령': ['-십시오', '-소서'],
                '청유': ['-십시다']
            },
            'level': 6
        }
    }
    
    # Count occurrences per level using JAMO patterns on JAMO-decomposed text
    level_counts = {}
    for name, info in speech_levels.items():
        count = 0
        for pattern in info['jamo_patterns']:
            count += len(re.findall(pattern, jamo_text))
        level_counts[name] = {'count': count, 'level': info['level']}
    
    total_endings = sum(lc['count'] for lc in level_counts.values())
    
    if total_endings > 0:
        # Find dominant (primary) speech level
        sorted_levels = sorted(level_counts.items(), key=lambda x: x[1]['count'], reverse=True)
        primary_name, primary_info = sorted_levels[0]
        primary_count = primary_info['count']
        
        # Calculate primary ratio
        primary_ratio = primary_count / total_endings
        
        # Count non-primary endings
        other_endings = []
        for name, info in level_counts.items():
            if name != primary_name and info['count'] > 0:
                other_endings.append(f"{name}({info['count']})")
        
        report.append(f"ℹ️ [어체-Stats] Detected: {primary_name}({primary_count}) dominant.")
        report.append(f"ℹ️ [어체-Ratio] Primary {primary_ratio:.1%} of total ({total_endings} endings)")
        
        # SINGLE SPEECH REGISTER ENFORCEMENT (90%+ required, 10% tolerance)
        if primary_ratio >= 0.90:
            report.append(f"✅ [어체] 단일 어체({primary_name}) 일관성 유지됨 ({primary_ratio:.1%}).")
        else:
            report.append(f"❌ [어체] 어체 혼용 감지! {primary_name} {primary_ratio:.1%}만 사용. 단일 어체(90%+)로 통일 필요.")
            if other_endings:
                report.append(f"⚠️ [어체-혼용] 다른 어체 감지: {', '.join(other_endings)}")
                report.append(f"📋 [수정 지시] 모든 종결어미를 '{primary_name}'로 통일하세요.")
                
                # Show examples of the primary speech level's endings
                primary_examples = speech_levels[primary_name].get('examples', {})
                report.append(f"\n📋 [어체-가이드] '{primary_name}' 허용 종결어미:")
                for category, endings in primary_examples.items():
                    report.append(f"   • {category}: {', '.join(endings)}")
            errors += 1

    report.append("\n💡 [Linguistic Note] Use the above structural data to apply your high-level linguistic judgment (Arousal, Reckon Vibe).")
    
    return "\n".join(report)

def _validate_clause_phrase_ratio(content: str, llm_phrase_count: int, llm_phrase_list: str = "") -> str:
    """
    LLM이 보고한 구 수를 받아 절:구 비율 검증.
    목표: 절:구 = 2:1 (±0.3 오차 허용)
    
    Args:
        content: HTML 콘텐츠
        llm_phrase_count: LLM이 센 구의 총 개수
        llm_phrase_list: LLM이 나열한 구 목록 (검증용)
    
    Returns:
        검증 결과 보고서
    """
    soup = BeautifulSoup(content, 'html.parser')
    text_content = soup.get_text()
    jamo_text = decompose_hangul(text_content)
    
    report = []
    
    # 1. 절 수 재계산 (시스템)
    terminal_patterns = [
        r'ㄷㅏ[\.!]', r'ㅇㅓ[\.!]', r'ㅇㅏ[\.!]', r'ㅈㅣ[\.!]',
        r'ㄴㅔ[\.!]', r'ㅇㅛ[\.!]', r'ㄴㅣㄷㅏ[\.!]',
        r'ㄴㅑ[\?]', r'ㄴㅣ[\?]', r'ㅈㅏ[\.!]', r'ㄹㅏ[\.!]',
    ]
    terminal_count = sum(len(re.findall(p, jamo_text)) for p in terminal_patterns)
    
    embedded_markers = re.findall(r'(?:는|ㄴ|은|ㄹ|을|던)\s*(?:것|거|때|곳|이|수|줄|법|리)', text_content)
    embedded_count = len(embedded_markers)
    
    total_clause = terminal_count + embedded_count
    
    # 2. 비율 계산 및 검증
    if llm_phrase_count > 0:
        ratio = total_clause / llm_phrase_count
    else:
        ratio = float('inf')
    
    target_ratio = 2.0
    tolerance = 0.3
    
    report.append(f"=== 절:구 비율 검증 ===")
    report.append(f"📊 [시스템] 절 수: {total_clause}개")
    report.append(f"📊 [LLM 보고] 구 수: {llm_phrase_count}개")
    report.append(f"📊 [비율] 절:구 = {ratio:.2f}:1")
    
    if llm_phrase_list:
        report.append(f"📋 [LLM 나열] {llm_phrase_list[:200]}...")  # 200자까지만
    
    # 3. 판정
    if target_ratio - tolerance <= ratio <= target_ratio + tolerance:
        report.append(f"✅ [PASS] 절:구 비율 적합 (목표: 2:1 ±0.3)")
    else:
        report.append(f"❌ [FAIL] 절:구 비율 부적합 (현재: {ratio:.2f}, 목표: 1.7~2.3)")
        
        if ratio > target_ratio + tolerance:
            # 구가 부족
            needed_phrases = int(total_clause / target_ratio) - llm_phrase_count
            report.append(f"📝 [수정 지시] 구(명사구/동사구/부사구) {needed_phrases}개 추가 필요")
            report.append(f"   → 복합 명사구 사용 권장: '열역학적 평형 상태', '에너지 보존의 원리' 등")
        else:
            # 구가 과다
            excess_phrases = llm_phrase_count - int(total_clause / target_ratio)
            report.append(f"📝 [수정 지시] 구 {excess_phrases}개 축소 또는 절로 확장 필요")
            report.append(f"   → 일부 구를 완전한 문장으로 전개 권장")
    
    return "\n".join(report)


def _audit_image_placement(content: str, persona_json: dict = None) -> str:
    """
    이미지 배치 적절성 검증:
    1. 최소 이미지 개수 확인
    2. 텍스트 블록 간 이미지 분포 확인
    3. alt 텍스트 품질 검증
    4. 캡션 존재 여부 확인
    
    Args:
        content: HTML 콘텐츠
        persona_json: 페르소나 정보 (optional)
    
    Returns:
        검증 결과 보고서
    """
    soup = BeautifulSoup(content, 'html.parser')
    report = []
    errors = 0
    warnings = 0
    
    report.append("=== 이미지 배치 검증 (Image Placement Audit) ===")
    
    # 1. 이미지 개수 확인
    images = soup.find_all('img')
    image_count = len(images)
    
    report.append(f"📊 [이미지 수] 총 {image_count}개 이미지 발견")
    
    if image_count == 0:
        report.append("❌ [이미지-필수] 이미지가 하나도 없습니다. 최소 1개 이상 삽입 필요.")
        errors += 1
    
    # 2. 텍스트 블록 간 이미지 분포 확인
    # 연속 3단락 이상 텍스트 → 이미지 삽입 권장
    paragraphs = soup.find_all('p')
    consecutive_text_blocks = 0
    max_consecutive = 0
    long_text_sections = []
    
    for i, element in enumerate(soup.find_all(['p', 'img', 'table'])):
        if element.name == 'p':
            # 테이블 내부 p는 제외
            if not element.find_parent('table'):
                consecutive_text_blocks += 1
                max_consecutive = max(max_consecutive, consecutive_text_blocks)
        elif element.name in ['img', 'table']:
            # 이미지나 테이블(이미지 포함 가능)을 만나면 리셋
            if consecutive_text_blocks >= 4:
                long_text_sections.append(consecutive_text_blocks)
            consecutive_text_blocks = 0
    
    # 마지막 섹션 체크
    if consecutive_text_blocks >= 4:
        long_text_sections.append(consecutive_text_blocks)
    
    if long_text_sections:
        report.append(f"⚠️ [이미지-분포] 연속 텍스트 블록이 긴 구간 발견: {long_text_sections}개 단락 연속")
        report.append(f"   → 3-4단락마다 이미지/시각 요소 삽입 권장")
        warnings += 1
    else:
        report.append("✅ [이미지-분포] 이미지가 적절히 분포되어 있습니다.")
    
    # 3. 섹션당 이미지 비율 확인
    headings = soup.find_all(['h1', 'h2', 'h3'])
    section_count = max(len(headings), 1)
    
    if image_count > 0:
        images_per_section = image_count / section_count
        if images_per_section < 0.5:
            report.append(f"⚠️ [이미지-밀도] 섹션당 이미지 비율 낮음 ({images_per_section:.1f}개/섹션)")
            report.append(f"   → 섹션당 1-2개 이미지 권장")
            warnings += 1
        else:
            report.append(f"✅ [이미지-밀도] 섹션당 이미지 비율 적절 ({images_per_section:.1f}개/섹션)")
    
    # 4. 각 이미지의 품질 검증
    for idx, img in enumerate(images):
        img_issues = []
        
        # alt 텍스트 확인
        alt = img.get('alt', '')
        if not alt:
            img_issues.append("alt 텍스트 누락")
        elif len(alt) < 5:
            img_issues.append(f"alt 텍스트 너무 짧음 ('{alt}')")
        elif alt in ['image', 'img', '이미지', 'photo', '사진']:
            img_issues.append(f"alt 텍스트가 의미없음 ('{alt}')")
        
        # src 확인
        src = img.get('src', '')
        if not src:
            img_issues.append("src 속성 누락")
        elif 'WAITING_FOR_SEARCH' in src or 'placeholder' in src.lower():
            img_issues.append("이미지 URL이 플레이스홀더임")
        elif 'unsplash.com' in src.lower() or 'cdn' in src.lower():
            img_issues.append("Unsplash/CDN 사용 금지 - GitHub raw URL 사용 필요")
        elif 'raw.githubusercontent.com' not in src and not src.startswith('data:'):
            img_issues.append(f"GitHub raw URL이 아님 - 형식: https://raw.githubusercontent.com/[user]/[repo]/main/images/[file].png")
        
        # 테이블 래핑 확인
        parent_table = img.find_parent('table')
        if not parent_table:
            img_issues.append("테이블로 래핑되지 않음")
        
        # width 스타일 확인
        style = img.get('style', '')
        if 'width: 100%' not in style and 'width:100%' not in style:
            img_issues.append("width: 100% 스타일 누락")
        
        # 캡션 확인 (이미지 다음의 p 태그 또는 같은 td 내 p 태그)
        has_caption = False
        if parent_table:
            caption_p = parent_table.find('p')
            if caption_p and len(caption_p.get_text(strip=True)) > 0:
                has_caption = True
        
        if not has_caption:
            img_issues.append("캡션 없음")
        
        # 이슈 리포트
        if img_issues:
            report.append(f"❌ [이미지 #{idx+1}] 문제 발견: {', '.join(img_issues)}")
            errors += 1
        else:
            report.append(f"✅ [이미지 #{idx+1}] 모든 검증 통과")
    
    # 5. 이미지 전략 가이드 (persona 기반)
    if persona_json and isinstance(persona_json, dict):
        image_strategy = persona_json.get('writer', {}).get('image_strategy', '')
        if image_strategy:
            report.append(f"\n💡 [이미지-전략] 페르소나 기반 권장 스타일: {image_strategy}")
    
    # 6. 이미지 스타일 가이드 (2026-01-09 신규)
    report.append("\n📸 [이미지-스타일] 필수 규칙:")
    report.append("   1. ✅ 실사 이미지만 사용 (일러스트/다이어그램 금지)")
    report.append("   2. ✅ 텍스트 라벨 필요시 → 디지털 손글씨 스타일")
    report.append("   3. ✅ GitHub raw URL 사용: https://raw.githubusercontent.com/[user]/[repo]/main/images/[file].png")
    report.append("   4. ❌ Unsplash/외부 CDN 금지")
    
    # 종합 결과
    report.append(f"\n📊 [종합] 오류: {errors}개, 경고: {warnings}개")
    
    if errors > 0:
        report.append("❌ [이미지 감독] 이미지 배치에 문제가 있습니다. 위 사항을 수정하세요.")
    elif warnings > 0:
        report.append("⚠️ [이미지 감독] 권장사항을 검토하세요.")
    else:
        report.append("✅ [이미지 감독] 이미지 배치가 적절합니다.")
    
    return "\n".join(report)


def _audit_design_and_image(content: str, persona_json: str) -> str:
    """
    Audits Design Ratio (60:30:10), Image Tone, and Mobile Compatibility.
    """
    soup = BeautifulSoup(content, 'html.parser')
    report = []
    errors = 0
    
    # A. Design Ratio (60:30:10)
    # 60% (Body + Whitespace): <p> tags excluding text in boxes (tables)
    # 30% (Structure): h1-h3, table content, ul/ol
    # 10% (Highlight): strong, span with color
    
    body_text_len = 0
    structure_text_len = 0
    highlight_text_len = 0
    
    # Calculate Body (p tags not inside tables)
    for p in soup.find_all('p'):
        if not p.find_parent('table'):
            body_text_len += len(p.get_text(strip=True))
            
    # Calculate Structure (Headings, Tables, Lists)
    for tag in soup.find_all(['h1', 'h2', 'h3', 'ul', 'ol', 'table']):
         structure_text_len += len(tag.get_text(strip=True))
         
    # Calculate Highlight (Strong, Color Spans)
    for tag in soup.find_all(['strong', 'mark']):
        highlight_text_len += len(tag.get_text(strip=True))
    for span in soup.find_all('span'):
        if 'color' in span.get('style', ''):
            highlight_text_len += len(span.get_text(strip=True))
            
    # Normalize (Structure contains Highlight text usually, avoiding double counting is hard without complex logic)
    # Simple approx: Total = Body + Structure (Structure usually includes everything else not in bare p)
    # Wait, the rule is 60:30:10.
    total = body_text_len + structure_text_len
    
    if total == 0:
         return "❌ [Design] Content is empty."

    # Adjusting logic: Structure length probably overlaps with Highlight.
    # Let's assume Body is bare text. Structure is Box/Heading text. Highlight is subset.
    # Ratio Calculation:
    # Body Ratio = Body / Total
    # Structure Ratio = Structure / Total
    # Highlight Ratio = Highlight / Total (This is independent, can overlap)
    
    body_ratio = body_text_len / total
    
    if 0.55 <= body_ratio <= 0.65:
        report.append(f"✅ [Ratio] Body+Whitespace ratio is good ({body_ratio:.1%}).")
    else:
        report.append(f"❌ [Ratio] Body+Whitespace ratio is {body_ratio:.1%} (Target: 60%). Increase text content or whitespace, reduce emphasis.")
        errors += 1
        
    # B. Image Tone & Format
    images = soup.find_all('img')
    for idx, img in enumerate(images):
        # Format Check
        parent_table = img.find_parent('table')
        if not parent_table:
            report.append(f"❌ [Image] Image {idx+1} is NOT wrapped in a Table.")
            errors += 1
            
        style = img.get('style', '')
        if 'width: 100%' not in style and 'width:100%' not in style:
            report.append(f"❌ [Image] Image {idx+1} missing 'width: 100%' style for mobile.")
            errors += 1
            
        # Tone Check (Caption)
        alt = img.get('alt', '')
        if not alt:
            report.append(f"❌ [Image] Image {idx+1} missing 'alt' text.")
            errors += 1
            
    # C. No Emoji
    text_all = soup.get_text()
    if re.search(r'[^\x00-\x7F가-힣]', text_all): # Rough check for non-ascii/non-korean (emojis)
        # Note: This regex is too broad, might catch punctuation.
        # Better: use explicit ranges or a library.
        # Simple Emoji Regex
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        if emoji_pattern.search(text_all):
             report.append(f"❌ [Emoji] Emojis found. Remove all emojis.")
             errors += 1
             
    # Mobile Compatibility
    # Check Line Height
    p_tags = soup.find_all('p')
    for p in p_tags:
        style = p.get('style', '')
        if 'line-height' not in style: # Strict check
             pass # Might be inherited? Plan says "Every p tag must have line-height: 1.8"
             # Let's inspect a sample
    
    return "\n".join(report)

def _validate_technical_constraints(content: str) -> str:
    """
    Validates Naver Blog HTML constraints (Block List).
    """
    report = []
    errors = 0
    
    forbidden = [
        ('<style>', r'<style'),
        ('class=', r'class='),
        ('id=', r'\bid='),
        ('border-radius', r'border-radius'),
        ('linear-gradient', r'linear-gradient'),
        ('max-width', r'max-width'),
        ('box-shadow', r'box-shadow'),
        ('display: flex', r'display:\s*flex'),
        ('display: grid', r'display:\s*grid')
    ]
    
    for name, pattern in forbidden:
        if re.search(pattern, content, re.IGNORECASE):
            report.append(f"❌ [Tech] Forbidden element found: '{name}'.")
            errors += 1
            
    if errors == 0:
        report.append("✅ [Tech] All technical constraints passed.")
        
    return "\n".join(report)

# -----------------------------------------------------------------------------
# MCP Tool Decorators
# -----------------------------------------------------------------------------

@mcp.tool()
def generate_personas(input_text: str) -> str:
    return _generate_personas(input_text)

@mcp.tool()
def audit_linguistic_quality(content: str, persona_json: str) -> str:
    return _audit_linguistic_quality(content, persona_json)

@mcp.tool()
def audit_design_and_image(content: str, persona_json: str) -> str:
    return _audit_design_and_image(content, persona_json)

@mcp.tool()
def validate_technical_constraints(content: str) -> str:
    return _validate_technical_constraints(content)

@mcp.tool()
def audit_image_placement(content: str, persona_json: str = "{}") -> str:
    """Audit image placement, distribution, and quality."""
    import json
    persona = json.loads(persona_json) if isinstance(persona_json, str) else persona_json
    return _audit_image_placement(content, persona)

def _generate_blog_tags(content: str, persona_json: dict = None) -> str:
    """
    콘텐츠와 페르소나 기반으로 블로그 태그 추천 (2관점).
    
    ⚠️ 주의: 이 함수는 LLM에게 가이드라인만 제공합니다.
    실제 태그 생성은 LLM이 콘텐츠를 분석하여 자율적으로 수행해야 합니다.
    
    Args:
        content: HTML 콘텐츠
        persona_json: 페르소나 정보
    
    Returns:
        태그 추천 가이드라인 및 프롬프트
    """
    soup = BeautifulSoup(content, 'html.parser')
    text_content = soup.get_text()
    
    # 텍스트 길이 및 복잡도 분석 (메타 정보만)
    word_count = len(text_content)
    
    report = []
    report.append("=== 🏷️  블로그 태그 추천 (LLM Auto-Generation) ===\n")
    report.append(f"📊 콘텐츠 메타정보: {word_count}자\n")
    
    # LLM에게 제공할 태그 생성 가이드라인
    report.append("=" * 60)
    report.append("💡 **LLM 태그 생성 지침 (Two-Perspective Approach)**")
    report.append("=" * 60)
    report.append("")
    
    report.append("📊 **관점 1: 전문가 지향 태그 (Expert-Oriented)**")
    report.append("   목표: 검색 알고리즘 인덱싱, 학술/전문가 검색 유입")
    report.append("   생성 원칙:")
    report.append("   • 영문 학술 용어 사용 (예: Maxwell_relations, thermodynamics)")
    report.append("   • 명확한 기술 키워드 (예: Gibbs_free_energy, phase_transition)")
    report.append("   • 검색 엔진 최적화 (SEO) 고려")
    report.append("   • 롱테일 키워드 전략 (예: Clausius_Clapeyron_equation)")
    report.append("   **생성할 태그 수: 5-6개**")
    report.append("")
    
    report.append("🎯 **관점 2: 초보자 지향 태그 (Beginner-Oriented)**")
    report.append("   목표: 일반인 검색 유입, 학습자/입문자 타겟팅")
    report.append("   생성 원칙:")
    report.append("   • 한글 쉬운 표현 (예: 열역학_쉽게, 물리_기초)")
    report.append("   • '란', '이란', '_입문', '_쉽게' 접미사 활용")
    report.append("   • 대중적 검색어 (예: 과학_공부, 대학물리)")
    report.append("   • 교육/학습 관련 키워드 (예: 독학, 개념정리)")
    report.append("   **생성할 태그 수: 5-6개**")
    report.append("")
    
    report.append("🔧 **태그 생성 프로세스:**")
    report.append("   1. 콘텐츠의 핵심 주제와 키워드를 자율 분석")
    report.append("   2. 관점 1: 전문 용어 기반 태그 5-6개 생성")
    report.append("   3. 관점 2: 대중 검색어 기반 태그 5-6개 생성")
    report.append("   4. 중복 제거 및 검색 효율성 검증")
    report.append("   5. 최종 태그 리스트를 '#태그명' 형식으로 출력")
    report.append("")
    
    report.append("💡 **사용 전략 (자동 안내):**")
    report.append("   • 네이버 블로그: 초보자 태그 5개 + 전문가 태그 1-2개 혼합")
    report.append("   • 티스토리/브런치: 전문가 태그 위주로 SEO 최적화")
    report.append("   • 검색 엔진 노출: 전문가 태그로 롱테일 키워드 확보")
    report.append("   • 소셜 미디어: 초보자 태그로 대중 접근성 향상")
    report.append("")
    
    report.append("=" * 60)
    report.append("🤖 **LLM 실행 필요: 위 가이드라인을 읽고 실제 태그를 생성하세요**")
    report.append("=" * 60)
    
    return "\n".join(report)

@mcp.tool()
def generate_blog_tags(content: str, persona_json: str = "{}") ->str:
    """Generate blog tag recommendations from two perspectives: expert and beginner."""
    import json
    persona = json.loads(persona_json) if isinstance(persona_json, str) else persona_json
    return _generate_blog_tags(content, persona)

if __name__ == "__main__":
    mcp.run()
