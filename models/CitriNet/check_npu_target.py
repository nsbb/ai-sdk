#!/usr/bin/env python3
"""
NPU 타겟 ID 확인
"""

# VeriSilicon NPU 타겟 ID 매핑
NPU_TARGETS = {
    # VIP9000 시리즈
    0x10000016: "VIP9000NANOSI_PLUS (실제 하드웨어)",
    0x10000020: "VIP9000 다른 변형 (시뮬레이터?)",
    
    # 일반적인 타겟들
    "T527": "Allwinner T527 (VIP9000PICO)",
    "VIP9000PICO_PID0XEE": "VIP9000PICO (T527 권장)",
    "VIP9000NANOSI_PLUS_PID0X10000016": "VIP9000NANOSI_PLUS",
}

print("=" * 70)
print("NPU 타겟 ID 정보")
print("=" * 70)
print()
print("현재 상황:")
print("  - NBG 파일에 저장된 타겟: 0x10000020")
print("  - 실제 하드웨어 타겟:      0x10000016")
print()
print("문제: 타겟 불일치로 인해 NBG 실행 실패")
print()
print("=" * 70)
print("해결 방법:")
print("=" * 70)
print()
print("1. 모든 중간 파일 삭제:")
print("   rm -rf npu_output/ wksp/ *.json *.data *.quantize")
print()
print("2. Quantize 단계에 --optimize 추가:")
print("   pegasus quantize \\")
print("     --optimize VIP9000NANOSI_PLUS_PID0X10000016 \\")
print("     ...")
print()
print("3. Export 단계에도 동일한 타겟 사용:")
print("   pegasus export ovxlib \\")
print("     --optimize VIP9000NANOSI_PLUS_PID0X10000016 \\")
print("     ...")
print()
print("=" * 70)
print("권장 타겟 (T527 기준):")
print("=" * 70)
print()
print("  Option 1: --optimize T527")
print("  Option 2: --optimize VIP9000PICO_PID0XEE")
print("  Option 3: --optimize VIP9000NANOSI_PLUS_PID0X10000016")
print()
print("현재 워크스페이스는 Option 1 (T527)을 사용합니다.")
print("=" * 70)

