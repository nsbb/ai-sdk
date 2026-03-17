#!/bin/bash
# Korean Wav2Vec2 XLS-R-300M 3s → T527 NPU uint8 NB conversion pipeline
# Usage: bash run_pipeline.sh [step]
#   step: import, quantize, export, all (default: all)

set -e

NAME="wav2vec2_ko_3s"
ACUITY_PATH="/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0/bin"
PEGASUS="$ACUITY_PATH/pegasus"
VIVANTE57="/home/nsbb/VeriSilicon/VivanteIDE5.7.2"
OPTIMIZE="VIP9000NANOSI_PLUS_PID0X10000016"  # T527

STEP=${1:-all}
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

echo "=== Korean Wav2Vec2 XLS-R-300M 3s Pipeline ==="
echo "Working directory: $WORKDIR"
echo "Step: $STEP"

do_import() {
    echo ""
    echo "=== Step 1: Import ONNX ==="
    # Remove old files
    rm -f ${NAME}.json ${NAME}.data

    $PEGASUS import onnx \
        --model ${NAME}.onnx \
        --output-model ${NAME}.json \
        --output-data ${NAME}.data

    if [ -f ${NAME}.json ] && [ -f ${NAME}.data ]; then
        echo "Import SUCCESS"
        ls -lh ${NAME}.json ${NAME}.data
    else
        echo "Import FAILED"
        exit 1
    fi

    # Generate inputmeta template (we'll use our own)
    echo "Using custom inputmeta: ${NAME}_inputmeta.yml"
}

do_quantize() {
    echo ""
    echo "=== Step 2: Quantize (uint8, moving_average) ==="

    # Update inputmeta with correct port name from imported model
    # First check the port name
    PORT_NAME=$(python3 -c "
import json
with open('${NAME}.json') as f:
    model = json.load(f)
for name, layer in model['Layers'].items():
    if layer.get('op', '') == 'input':
        print(name)
        break
" 2>/dev/null || echo "input")
    echo "Detected input port: $PORT_NAME"

    # Create proper inputmeta with correct port name
    cat > ${NAME}_inputmeta.yml << HEREDOC
# !!!This file disallow TABs!!!
input_meta:
  databases:
  - path: dataset.txt
    type: TEXT
    ports:
    - lid: ${PORT_NAME}
      category: frequency
      dtype: float32
      sparse: false
      tensor_name:
      shape:
      - 1
      - 48000
      fitting: scale
      preprocess:
        reverse_channel: false
        scale: 1.0
        preproc_node_params:
          add_preproc_node: false
          preproc_type: TENSOR
          preproc_perm:
          - 0
          - 1
      redirect_to_output: false
HEREDOC

    echo "Inputmeta created with reverse_channel: false"

    rm -f ${NAME}_uint8.quantize

    $PEGASUS quantize \
        --model ${NAME}.json \
        --model-data ${NAME}.data \
        --device CPU \
        --with-input-meta ${NAME}_inputmeta.yml \
        --rebuild-all \
        --algorithm moving_average \
        --moving-average-weight 0.004 \
        --quantizer asymmetric_affine \
        --qtype uint8 \
        --model-quantize ${NAME}_uint8.quantize

    if [ -f ${NAME}_uint8.quantize ]; then
        echo "Quantize SUCCESS"
        ls -lh ${NAME}_uint8.quantize
    else
        echo "Quantize FAILED"
        exit 1
    fi
}

do_export() {
    echo ""
    echo "=== Step 3: Export NB (Docker) ==="

    ACUITY612="/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0"

    docker run --rm \
        -v "$WORKDIR":/work \
        -v "$VIVANTE57":/vivante57:ro \
        -v "$ACUITY612":/acuity612:ro \
        t527-npu:v1.2 \
        bash -c "
            VSIM=/vivante57/cmdtools/vsimulator
            COMMON=/vivante57/cmdtools/common
            export REAL_GCC=/usr/bin/gcc
            export VIVANTE_VIP_HOME=/vivante57
            export VIVANTE_SDK_DIR=\$VSIM
            export LD_LIBRARY_PATH=\$VSIM/lib:\$COMMON/lib:\$VSIM/lib/x64_linux:\$VSIM/lib/x64_linux/vsim:\$LD_LIBRARY_PATH
            export EXTRALFLAGS=\"-Wl,--disable-new-dtags -Wl,-rpath,\$VSIM/lib -Wl,-rpath,\$COMMON/lib -Wl,-rpath,\$VSIM/lib/x64_linux -Wl,-rpath,\$VSIM/lib/x64_linux/vsim\"
            cd /acuity612/bin
            ./pegasus export ovxlib \
                --model /work/${NAME}.json \
                --model-data /work/${NAME}.data \
                --dtype quantized \
                --model-quantize /work/${NAME}_uint8.quantize \
                --with-input-meta /work/${NAME}_inputmeta.yml \
                --pack-nbg-unify \
                --optimize ${OPTIMIZE} \
                --viv-sdk \$VSIM \
                --target-ide-project linux64 \
                --batch-size 1 \
                --output-path /work/wksp/${NAME}_uint8/
        "

    NB_PATH="wksp/${NAME}_uint8_nbg_unify/network_binary.nb"
    if [ -f "$NB_PATH" ]; then
        NB_SIZE=$(ls -lh "$NB_PATH" | awk '{print $5}')
        echo "Export SUCCESS: $NB_PATH ($NB_SIZE)"
    else
        echo "Export FAILED - network_binary.nb not found"
        ls -la wksp/ 2>/dev/null
        exit 1
    fi
}

case $STEP in
    import)    do_import ;;
    quantize)  do_quantize ;;
    export)    do_export ;;
    all)
        do_import
        do_quantize
        do_export
        ;;
    *)
        echo "Unknown step: $STEP (use: import, quantize, export, all)"
        exit 1
        ;;
esac

echo ""
echo "=== Pipeline Complete ==="
