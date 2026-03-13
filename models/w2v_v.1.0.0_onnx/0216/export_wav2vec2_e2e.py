import torch
from transformers import Wav2Vec2ForCTC

class Wav2Vec2E2E(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def forward(self, x):
        # x: [B, T]  (B가 time folding용)
        out = self.model(x)
        return out.logits

if __name__ == "__main__":
    model = Wav2Vec2E2E()
    model.eval()

    # ⚠️ 핵심: T를 B로 접기
    # 예: 1초 오디오 (16000) → (16000, 1)
    dummy = torch.randn(16000, 1)

    torch.onnx.export(
        model,
        dummy,
        "../2_onnx/wav2vec2_e2e.onnx",
        input_names=["waveform"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
    )

    print("[OK] wav2vec2 end-to-end ONNX exported")

