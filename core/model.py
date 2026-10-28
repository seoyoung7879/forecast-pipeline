from pathlib import Path
import joblib
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "모델"


class ResBlock(nn.Module):
    def __init__(s, ch, dil, drop):
        super().__init__()
        s.conv = nn.Conv1d(ch, 2*ch, 3, padding=dil, dilation=dil)
        s.res, s.skip = nn.Conv1d(ch, ch, 1), nn.Conv1d(ch, ch, 1)
        s.drop = nn.Dropout(drop)
    def forward(s, x):
        a, b = s.conv(x).chunk(2, dim=1)
        h = s.drop(torch.tanh(a) * torch.sigmoid(b))
        return x + s.res(h), s.skip(h)


class Net(nn.Module):
    def __init__(s, n_in, ch=48, drop=0.2, dils=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        s.inp = nn.Conv1d(n_in, ch, 1)
        s.blocks = nn.ModuleList([ResBlock(ch, d, drop) for d in dils])
        s.head = nn.Sequential(nn.ReLU(), nn.Conv1d(ch, ch, 1), nn.ReLU(), nn.Conv1d(ch, 1, 1))
    def forward(s, x):
        h, skip = s.inp(x), 0
        for b in s.blocks:
            h, sk = b(h); skip = skip + sk
        return s.head(skip).squeeze(1)


def activate(raw, task, day):
    return torch.sigmoid(raw) if task == "eff" else F.softplus(raw) * day.float()


def _load(path):
    b = joblib.load(path)
    nets = []
    for sd_ in b["state_dicts"]:
        n = Net(**b["cfg"])
        n.load_state_dict(sd_)
        n.eval()
        nets.append(n)
    b["nets"] = nets
    return b


_CACHE = {}

def load_models():
    if not _CACHE:
        _CACHE["irr"] = _load(MODEL_DIR / "cnn_최종_일사.joblib")
        _CACHE["eff"] = _load(MODEL_DIR / "cnn_최종_효율.joblib")
    return _CACHE["irr"], _CACHE["eff"]


@torch.no_grad()
def ensemble(nets, X, day, task, bs=2048):
    X, day = torch.as_tensor(X), torch.as_tensor(day)
    out = []
    for i in range(0, len(X), bs):
        xb, db = X[i:i + bs], day[i:i + bs]
        out.append(torch.stack([activate(n(xb), task, db) for n in nets]).mean(0))
    return torch.cat(out).numpy()