from pathlib import Path
import joblib
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "모델"


class Res(nn.Module):
    def __init__(s, c, drop):
        super().__init__()
        s.c1, s.c2 = nn.Conv1d(c, c, 3, padding=1), nn.Conv1d(c, c, 3, padding=1)
        s.n1, s.n2 = nn.GroupNorm(1, c), nn.GroupNorm(1, c)
        s.dp = nn.Dropout(drop)

    def forward(s, x):
        h = s.dp(F.gelu(s.n1(s.c1(x))))
        return F.gelu(x + s.n2(s.c2(h)))


class Net(nn.Module):
    def __init__(s, n_in, width=64, nblk=4, drop=0.2):
        super().__init__()
        s.stem = nn.Conv1d(n_in, width, 1)
        s.blocks = nn.Sequential(*[Res(width, drop) for _ in range(nblk)])
        s.head = nn.Conv1d(width, 1, 1)

    def forward(s, x):
        return s.head(s.blocks(s.stem(x))).squeeze(1)


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
def ensemble(nets, X, bs=2048):
    X = torch.as_tensor(X)
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        out.append(torch.stack([n(xb) for n in nets]).mean(0))
    return torch.cat(out).numpy()