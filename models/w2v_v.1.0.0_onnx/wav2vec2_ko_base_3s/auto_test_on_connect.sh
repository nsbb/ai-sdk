#!/bin/bash
# Auto-detect device and run priority NB tests when T527 connects
# Usage: nohup bash auto_test_on_connect.sh &

WIN_ADB="/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${WORK_DIR}/auto_test_log.txt"

echo "=== Auto Test Watchdog ===" | tee "$LOG"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "Waiting for device to connect..." | tee -a "$LOG"

while true; do
    # Check device
    if $WIN_ADB devices 2>/dev/null | grep -q "device$"; then
        echo "" | tee -a "$LOG"
        echo "=== DEVICE FOUND! $(date '+%H:%M:%S') ===" | tee -a "$LOG"

        # Wait for device to fully initialize
        sleep 3

        # Run priority tests
        echo "Running priority NB tests..." | tee -a "$LOG"
        bash "${WORK_DIR}/test_priority_nbs.sh" 2>&1 | tee -a "$LOG"

        echo "" | tee -a "$LOG"
        echo "=== Running split model test (CNN uint8 + Transformer int16)... ===" | tee -a "$LOG"
        if [ -f "${WORK_DIR}/test_split_model.sh" ]; then
            bash "${WORK_DIR}/test_split_model.sh" 2>&1 | tee -a "$LOG"
        fi

        echo "" | tee -a "$LOG"
        echo "=== Priority tests complete. Running XLS-R test... ===" | tee -a "$LOG"

        # Run XLS-R test if available
        XLSR_DIR="${WORK_DIR}/../wav2vec2_ko_xls_r_300m_3s"
        if [ -f "${XLSR_DIR}/test_xlsr_opset12_nb.sh" ]; then
            bash "${XLSR_DIR}/test_xlsr_opset12_nb.sh" 2>&1 | tee -a "$LOG"
        fi

        echo "" | tee -a "$LOG"
        echo "=== ALL TESTS COMPLETE $(date '+%H:%M:%S') ===" | tee -a "$LOG"
        echo "Results saved to: $LOG"
        echo "Priority results: ${WORK_DIR}/npu_test_results.txt"

        exit 0
    fi

    # Wait 30 seconds before checking again
    sleep 30
done
