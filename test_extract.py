import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('소화전.csv', encoding='utf-8-sig')
col0 = df.columns[0]

_KNOWN_SIGUNGU = [
    '수원시','성남시','의정부시','안양시','부천시','광명시','평택시','동두천시','안산시',
    '고양시','과천시','구리시','남양주시','오산시','시흥시','군포시','의왕시','하남시',
    '용인시','파주시','이천시','안성시','김포시','화성시','광주시','양주시','포천시',
    '여주시','연천군','가평군','양평군',
]

def extract_sigungu(addr):
    if not isinstance(addr, str):
        return '기타'
    addr = addr.strip()
    parts = addr.split()
    if len(parts) < 2:
        return '기타'
    raw_token = parts[1]
    if raw_token.endswith(('시','군','구')):
        for known in _KNOWN_SIGUNGU:
            if raw_token == known:
                return known
            if len(raw_token) >= 3 and len(known) >= 3:
                if raw_token[-1] == known[-1] and sum(a==b for a,b in zip(raw_token, known)) >= len(known)-1:
                    return known
        return raw_token
    for known in _KNOWN_SIGUNGU:
        if raw_token.startswith(known):
            return known
    return raw_token

# problematic samples
samples = [
    '경기도 냠양주시 진건오남로 759번길 11-3',
    '경기도 시흥시경기도과기대로 246',
]
print('Original -> Extracted')
for s in samples:
    print(f"{s} -> {extract_sigungu(s)}")
