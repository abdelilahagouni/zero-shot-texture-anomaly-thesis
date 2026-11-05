"""
Shared code for the self-contamination mechanism experiments (mechanism_real.py, mechanism_synthetic.py).

Token pipeline replicates src/eval_router_full.py: DINOv2 ViT-B/14 at 518x518, L2-normalised patch tokens
cached in fp16 on CPU, computed in fp32 on GPU, PyTorch default TF32 settings.

Route A        : s_i = 1 - mean_j cos(x_i, x_j) over ALL patches j (existing).
Route A-clean  : s_i = 1 - mean_j cos(x_i, x_j) over NON-defect patches j only (uses the ground truth;
                 diagnostic, not deployable). A patch is a defect patch if any pixel of its 14x14 window
                 overlaps the mask. Defect patches are scored against the clean patches too.
Route A-median : s_i = 1 - median_j cos(x_i, x_j) over all patches.
Route A-trim   : s_i = 1 - mean of cos(x_i, x_j) after dropping the lowest 25% of similarities.
Route B        : s_i = 1 - max_m cos(x_i, m) over a bank of 4 training images,
                 chosen by np.random.default_rng(seed).choice(n_train, 4, replace=False).
Post-processing (all): bilinear 37x37 -> 518x518 (align_corners=False) + cv2.GaussianBlur 15x15.
Contamination index: 1 - cos(mean token of all patches, mean token of non-defect patches).
"""
import numpy as np

IMG, GRID, PATCH = 518, 37, 14
A_METHODS = ["A", "A_clean", "A_median", "A_trim"]


class Backbone:
    def __init__(self):
        import torch, torch.nn.functional as F
        from torchvision import transforms
        self.torch, self.F = torch, F
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(self.dev).eval()
        self.tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def tokens(self, pil):
        """fp32 GPU tokens after an fp16 round trip (as eval_router_full.py caches them)."""
        img = self.tf(pil.convert("RGB")).unsqueeze(0).to(self.dev)
        with self.torch.no_grad():
            f = self.model.forward_features(img)["x_norm_patchtokens"][0]
        f = self.F.normalize(f, p=2, dim=-1).cpu().to(self.torch.float16)
        return f.to(self.dev).float()

    def post(self, s):
        import cv2
        hm = self.F.interpolate(s.reshape(1, 1, GRID, GRID), size=(IMG, IMG), mode="bilinear",
                                align_corners=False).squeeze().cpu().numpy()
        return cv2.GaussianBlur(hm, (15, 15), 0)

    def bank(self, train_tokens, seed, k=4):
        idx = np.random.default_rng(seed).choice(len(train_tokens), size=k, replace=False)
        return self.torch.cat([train_tokens[i] for i in idx], dim=0)

    def maps(self, tok, mask, banks):
        """Returns ({method: 518x518 map or None}, contamination index). mask: 518x518 bool."""
        torch = self.torch
        dp = patch_defect_mask(mask)
        clean = torch.from_numpy(~dp).to(self.dev)
        sim = tok @ tok.T
        out = {"A": self.post(1.0 - sim.mean(dim=1))}
        if clean.any():
            out["A_clean"] = self.post(1.0 - sim[:, clean].mean(dim=1))
            mu_all, mu_clean = tok.mean(0), tok[clean].mean(0)
            contam = float(1.0 - self.F.cosine_similarity(mu_all, mu_clean, dim=0).item())
        else:
            out["A_clean"], contam = None, float("nan")
        out["A_median"] = self.post(1.0 - sim.median(dim=1).values)
        srt = torch.sort(sim, dim=1).values
        out["A_trim"] = self.post(1.0 - srt[:, int(np.floor(0.25 * sim.shape[1])):].mean(dim=1))
        for sd, bk in banks.items():
            out[f"B{sd}"] = self.post(1.0 - (tok @ bk.T).max(dim=1)[0])
        return out, contam


def patch_defect_mask(mask):
    """37x37 -> flattened (row-major, token order) bool: patch overlaps the mask anywhere."""
    m = np.asarray(mask, bool).reshape(GRID, PATCH, GRID, PATCH).any(axis=(1, 3))
    return m.ravel()


def fe_solve(x_cols, Y, groups):
    """OLS with group fixed effects (one dummy per group, no intercept).
    x_cols: (n, p) regressors of interest; Y: (n,) or (n, q). Returns coefficients on x_cols: (p,) or (p, q)."""
    g = np.asarray(groups)
    levels = np.unique(g)
    D = (g[:, None] == levels[None, :]).astype(float)
    X = np.column_stack([np.atleast_2d(np.asarray(x_cols, float).T).T, D])
    beta, *_ = np.linalg.lstsq(X, np.asarray(Y, float), rcond=None)
    p = np.atleast_2d(np.asarray(x_cols, float).T).T.shape[1]
    return beta[:p]


def ci95(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
