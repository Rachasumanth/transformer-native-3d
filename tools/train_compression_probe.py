"""Single-fixture CUDA optimization check, not dataset training or a quality benchmark."""
import json
from pathlib import Path
import numpy as np
import torch
from t3d.container import load
from t3d.sc_vae import SparseCompressionVAE, reconstruction_loss


def main():
    assert torch.cuda.is_available(), 'Kaggle GPU required'
    torch.manual_seed(8)
    root = Path(__file__).resolve().parents[1]
    asset = load(root/'artifacts/roundtrip/cube.t3d').asset
    coords = torch.tensor(np.column_stack((np.zeros(len(asset.coords), dtype=int), asset.coords)), dtype=torch.int64, device='cuda')
    shape = np.column_stack((asset.dual_vertices, asset.intersected_flag, np.zeros((len(asset.coords), 1))))
    results = {}
    for mode, values in [('shape', shape), ('material', asset.pbr)]:
        features = torch.tensor(values, dtype=torch.float32, device='cuda')
        model = SparseCompressionVAE(mode, channels=(4, 8, 16, 32, 64), latent_channels=8).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        history = []
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            # Posterior mean reduces variance in this deterministic execution check.
            loss, terms = reconstruction_loss(model, coords, features, stochastic=False)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            history.append(dict(step=step, total=loss.item(), **{k:v.item() for k,v in terms.items()}))
        torch.save(dict(mode=mode, channels=(4,8,16,32,64), latent_channels=8,
                        model=model.state_dict(), optimizer=optimizer.state_dict(), steps=10),
                   root.parent/f'{mode}-vae-probe.pt')
        results[mode] = history
    report = dict(fixture='8^3 cube with synthetic constant source material', device=torch.cuda.get_device_name(0),
                  scope='single-fixture deterministic posterior-mean optimization; no generalization claim', results=results)
    (root.parent/'compression-probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({mode:dict(initial=h[0]['total'],final=h[-1]['total']) for mode,h in results.items()}))


if __name__ == '__main__':
    main()
