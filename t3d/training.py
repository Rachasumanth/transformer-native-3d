"""Stateful VAE optimization and resumable checkpoints for prepared asset batches."""
import os
import math
import random
from pathlib import Path
import tempfile
import torch
from .dataset import batch_arrays
from .sc_vae import SparseCompressionVAE, reconstruction_loss


class VAETrainer:
    def __init__(self, *, mode, channels=(32,64,128,256,512), latent_channels=16,
                 device='cuda', learning_rate=1e-4, kl_weight=1e-6, manifest_sha256='',edge_positive_weight=None,
                 render_weight=0., convolution_backend='reference', occupancy_level_weights=None,
                 shuffle_seed=None, batch_size=None):
        self.config = dict(mode=mode, channels=tuple(channels), latent_channels=latent_channels)
        if convolution_backend != 'reference':
            self.config['convolution_backend'] = convolution_backend
        self.model = SparseCompressionVAE(**self.config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=.1)
        self.kl_weight, self.manifest_sha256 = kl_weight, manifest_sha256
        self.edge_positive_weight=edge_positive_weight
        if occupancy_level_weights is not None:
            occupancy_level_weights=tuple(float(x) for x in occupancy_level_weights)
            if len(occupancy_level_weights)!=4 or not all(math.isfinite(x) and x>=0 for x in occupancy_level_weights) or sum(occupancy_level_weights)<=0:
                raise ValueError('Occupancy level weights must be four finite nonnegative values with positive sum')
        self.occupancy_level_weights=occupancy_level_weights
        if render_weight<0 or (render_weight and mode!='shape'):raise ValueError('render supervision requires shape mode and nonnegative weight')
        self.render_weight=render_weight
        if shuffle_seed is not None and (type(shuffle_seed) is not int or shuffle_seed < 0):
            raise ValueError('Shuffle seed must be a nonnegative integer')
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError('Batch size must be a positive integer')
        self.shuffle_seed, self.batch_size = shuffle_seed, batch_size
        self._order_cache_key, self._order_cache = None, None
        self.steps, self.epoch, self.batch_cursor = 0, 0, 0

    def next_indices(self, dataset_size, batch_size):
        """Return the next epoch slice without consuming model RNG state."""
        if type(dataset_size) is not int or dataset_size <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('Dataset and batch sizes must be positive integers')
        if self.batch_size is not None and batch_size != self.batch_size:
            raise ValueError('Batch size differs from checkpoint')
        if self.batch_cursor >= dataset_size:
            self.epoch += 1
            self.batch_cursor = 0
        end=min(dataset_size,self.batch_cursor+batch_size)
        if self.shuffle_seed is None:
            return list(range(self.batch_cursor,end))
        key=(dataset_size,self.shuffle_seed,self.epoch)
        if key != self._order_cache_key:
            order=list(range(dataset_size))
            random.Random(f'{self.shuffle_seed}:{self.epoch}').shuffle(order)
            self._order_cache_key,self._order_cache=key,order
        return self._order_cache[self.batch_cursor:end]

    def batch(self, assets):
        coords, features = batch_arrays(assets, self.config['mode'])
        device = next(self.model.parameters()).device
        return torch.from_numpy(coords).to(device), torch.from_numpy(features).to(device)

    def update(self, assets):
        self.model.train()
        coords, features = self.batch(assets)
        self.optimizer.zero_grad(set_to_none=True)
        render_fn=self._render_fn(assets)
        loss, parts = reconstruction_loss(self.model, coords, features, kl_weight=self.kl_weight,
                                           edge_positive_weight=self.edge_positive_weight,render_fn=render_fn,
                                           render_weight=self.render_weight,occupancy_level_weights=self.occupancy_level_weights)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('Nonfinite training gradients')
        self.optimizer.step()
        self.steps += 1
        self.batch_cursor += len(assets)
        return dict(total=loss.item(), **{k:v.item() for k,v in parts.items()})

    @torch.no_grad()
    def evaluate(self, assets):
        self.model.eval()
        coords, features = self.batch(assets)
        loss, parts = reconstruction_loss(self.model, coords, features, stochastic=False, kl_weight=self.kl_weight,
                                           edge_positive_weight=self.edge_positive_weight,render_fn=self._render_fn(assets),
                                           render_weight=self.render_weight,occupancy_level_weights=self.occupancy_level_weights)
        return dict(total=loss.item(), **{k:v.item() for k,v in parts.items()})

    def _render_fn(self,assets):
        if not self.render_weight:return None
        if len(assets)!=1:raise ValueError('render supervision currently requires batch size 1')
        from .training_render import shape_supervision
        return lambda coords,target,raw:shape_supervision(assets[0],coords,target,raw)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(version=1, config=self.config, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     kl_weight=self.kl_weight, manifest_sha256=self.manifest_sha256,
                     edge_positive_weight=self.edge_positive_weight,
                     render_weight=self.render_weight, occupancy_level_weights=self.occupancy_level_weights,
                     shuffle_seed=self.shuffle_seed, batch_size=self.batch_size,
                     steps=self.steps, epoch=self.epoch, batch_cursor=self.batch_cursor,
                     rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
        os.close(fd)
        try:
            with open(temporary, 'wb') as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def resume(cls, path, *, device='cuda', manifest_sha256):
        # Load only trusted project checkpoints; torch serialization includes optimizer state.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['version'] != 1 or state['manifest_sha256'] != manifest_sha256:
            raise ValueError('Checkpoint version or dataset fingerprint mismatch')
        trainer = cls(**state['config'], device=device, kl_weight=state['kl_weight'], manifest_sha256=manifest_sha256,
                      edge_positive_weight=state.get('edge_positive_weight'),render_weight=state.get('render_weight',0.),
                      occupancy_level_weights=state.get('occupancy_level_weights'),
                      shuffle_seed=state.get('shuffle_seed'),batch_size=state.get('batch_size'))
        trainer.model.load_state_dict(state['model'])
        trainer.optimizer.load_state_dict(state['optimizer'])
        trainer.steps, trainer.epoch, trainer.batch_cursor = state['steps'], state['epoch'], state['batch_cursor']
        torch.set_rng_state(state['rng_cpu'])
        if state['rng_cuda']:
            if not torch.cuda.is_available() or len(state['rng_cuda']) != torch.cuda.device_count():
                raise ValueError('Exact resume requires matching CUDA device count')
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        return trainer


def main():
    import argparse
    import json
    from .dataset import AssetDataset
    parser = argparse.ArgumentParser(description='Train a sparse VAE on prepared .t3d assets')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--mode', choices=['shape','material'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, required=True, help='Total optimizer steps including resumed steps')
    parser.add_argument('--batch-size', type=int, default=None, help='New-run batch size (default 1); resume restores checkpoint value')
    parser.add_argument('--shuffle-seed', type=int, default=None, help='Enable deterministic per-epoch shuffling')
    parser.add_argument('--occupancy-level-weights', nargs=4, type=float, default=None,
                        help='Optional coarse-to-fine occupancy loss weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--convolution-backend', choices=['reference','triton'], default=None,
                        help='Experimental float32 CUDA Triton option; default reference')
    args = parser.parse_args()
    if args.steps <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error('Steps and batch size must be positive')
    train = AssetDataset(args.manifest, split='train', require_pbr=args.mode=='material')
    validation = AssetDataset(args.manifest, split='validation', require_pbr=args.mode=='material')
    if not len(train) or not len(validation):
        raise ValueError('Training and held-out validation splits must both be nonempty')
    trainer = (VAETrainer.resume(args.checkpoint, device=args.device, manifest_sha256=train.fingerprint)
               if args.resume else VAETrainer(mode=args.mode, device=args.device, manifest_sha256=train.fingerprint,
                               convolution_backend=args.convolution_backend or 'reference',
                               occupancy_level_weights=args.occupancy_level_weights,
                               shuffle_seed=args.shuffle_seed,batch_size=args.batch_size or 1))
    if args.resume and args.convolution_backend is not None and args.convolution_backend != trainer.config.get('convolution_backend','reference'):
        raise ValueError('Requested backend differs from checkpoint')
    if args.resume and args.occupancy_level_weights is not None and tuple(args.occupancy_level_weights) != trainer.occupancy_level_weights:
        raise ValueError('Requested occupancy weights differ from checkpoint')
    if args.resume and args.shuffle_seed is not None and args.shuffle_seed != trainer.shuffle_seed:
        raise ValueError('Requested shuffle seed differs from checkpoint')
    if args.resume and args.batch_size is not None and trainer.batch_size is not None and args.batch_size != trainer.batch_size:
        raise ValueError('Requested batch size differs from checkpoint')
    batch_size=args.batch_size or trainer.batch_size or 1
    if trainer.config['mode'] != args.mode:
        raise ValueError('Requested mode differs from checkpoint')
    while trainer.steps < args.steps:
        indices = trainer.next_indices(len(train), batch_size)
        metrics = trainer.update([train[i] for i in indices])
        print(json.dumps(dict(step=trainer.steps, train=metrics)), flush=True)
        trainer.save(args.checkpoint)
    heldout = [trainer.evaluate([validation[i]]) for i in range(len(validation))]
    print(json.dumps(dict(validation_asset_mean={k:sum(m[k] for m in heldout)/len(heldout) for k in heldout[0]})))


if __name__ == '__main__':
    main()
