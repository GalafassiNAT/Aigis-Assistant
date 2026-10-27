import torch
import torch.nn as nn
from infer_rvc_python.lib.infer_pack.models import SynthesizerTrnMs768NSFsid

print("[1/3] Carregando pesos do modelo da Aigis...")
cpt = torch.load(
    "models/rvc/aigis/AigisMrM0dZ_300e_5700s.pth", map_location="cpu", weights_only=False
)
cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=False)
del net_g.enc_q
net_g.load_state_dict(cpt["weight"], strict=False)
net_g.eval()


class ExportWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, feats, p_len, pitch, pitchf, sid):
        out = self.net.infer(feats, p_len, pitch, pitchf, sid)
        return out[0]


wrapper = ExportWrapper(net_g)
wrapper.eval()

print("[2/3] Preparando tensores de teste...")
t_len = 50
feats = torch.randn(1, t_len, 768)
p_len = torch.tensor([t_len]).long()
pitch = torch.randint(0, 255, (1, t_len)).long()
pitchf = torch.randn(1, t_len).float()
sid = torch.tensor([0]).long()

output_onnx = "models/rvc/aigis/aigis_model.onnx"
print(f"[3/3] Exportando ONNX para: {output_onnx} (opset 16)...")

try:
    torch.onnx.export(
        wrapper,
        (feats, p_len, pitch, pitchf, sid),
        output_onnx,
        input_names=["feats", "p_len", "pitch", "pitchf", "sid"],
        output_names=["audio"],
        dynamic_axes={
            "feats": {1: "time_frames"},
            "pitch": {1: "time_frames"},
            "pitchf": {1: "time_frames"},
            "audio": {2: "samples"},
        },
        opset_version=16,
        do_constant_folding=True,
    )
    print("SUCCESS: Export concluído!")
except Exception as e:
    import traceback

    print(f"ERROR: {e}")
    traceback.print_exc()
