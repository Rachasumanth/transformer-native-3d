"""Bounded core GLB accessor decoding for rig ingestion; no scene evaluation."""
import json
from pathlib import Path
import struct
import numpy as np

_DTYPES = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


class GLBAccessors:
    def __init__(self, path):
        data = Path(path).read_bytes()
        if len(data) < 20 or struct.unpack_from('<4sII', data) != (b'glTF', 2, len(data)):
            raise ValueError('Invalid GLB header')
        chunks = []; offset = 12
        while offset < len(data):
            if offset+8 > len(data): raise ValueError('Truncated chunk header')
            size, kind = struct.unpack_from('<II', data, offset); offset += 8
            if size % 4 or offset+size > len(data): raise ValueError('Invalid chunk length')
            chunks.append((kind, data[offset:offset+size])); offset += size
        if not chunks or chunks[0][0] != 0x4e4f534a: raise ValueError('First chunk must be JSON')
        self.document = json.loads(chunks[0][1])
        buffers = self.document.get('buffers', [])
        binary = [b for kind, b in chunks if kind == 0x004e4942]
        if len(buffers) != 1 or buffers[0].get('uri') or len(binary) != 1:
            raise ValueError('One embedded binary buffer required')
        size = buffers[0]['byteLength']
        if size < 0 or not 0 <= len(binary[0])-size <= 3: raise ValueError('Invalid buffer size')
        self.binary = binary[0][:size]

    def _rows(self, view_index, offset, count, dtype, width, *, sparse=False):
        views = self.document.get('bufferViews', [])
        if not 0 <= view_index < len(views): raise ValueError('Invalid bufferView')
        view = views[view_index]; dtype = np.dtype(dtype)
        start, length = view.get('byteOffset', 0), view['byteLength']
        stride = view.get('byteStride', width*dtype.itemsize)
        if (view.get('buffer', 0) != 0 or start < 0 or length < 0 or start+length > len(self.binary)
                or offset < 0 or count <= 0 or stride < width*dtype.itemsize
                or stride % dtype.itemsize or (start+offset) % dtype.itemsize
                or offset+(count-1)*stride+width*dtype.itemsize > length
                or (sparse and 'byteStride' in view)):
            raise ValueError('Invalid accessor buffer span, stride or alignment')
        return np.ndarray((count, width), dtype=dtype, buffer=self.binary,
                          offset=start+offset, strides=(stride, dtype.itemsize)).copy()

    def read(self, index):
        accessors = self.document.get('accessors', [])
        if not 0 <= index < len(accessors): raise ValueError('Invalid accessor index')
        a = accessors[index]
        if a.get('extensions') or a['type'] not in _WIDTHS or a['componentType'] not in _DTYPES:
            raise ValueError('Unsupported accessor type or extension')
        if a['type'] == 'MAT4' and a['componentType'] != 5126:
            raise ValueError('Only float32 matrix accessors supported')
        count, width = a['count'], _WIDTHS[a['type']]
        if count <= 0 or count*width > 50000000: raise ValueError('Accessor allocation limit')
        dtype = _DTYPES[a['componentType']]
        if 'bufferView' in a:
            value = self._rows(a['bufferView'], a.get('byteOffset', 0), count, dtype, width)
        else:
            if a.get('byteOffset', 0): raise ValueError('Offset without bufferView')
            value = np.zeros((count, width), dtype=dtype)
        if 'sparse' in a:
            s = a['sparse']; ids = s['indices']; val = s['values']; n = s['count']
            if not 0 < n <= count or ids['componentType'] not in (5121, 5123, 5125):
                raise ValueError('Invalid sparse accessor count or indices')
            indices = self._rows(ids['bufferView'], ids.get('byteOffset', 0), n,
                                 _DTYPES[ids['componentType']], 1, sparse=True)[:, 0].astype(np.int64)
            if (indices >= count).any() or (np.diff(indices) <= 0).any():
                raise ValueError('Sparse indices must be increasing and in bounds')
            value[indices] = self._rows(val['bufferView'], val.get('byteOffset', 0), n, dtype, width, sparse=True)
        if a.get('normalized', False):
            if a['componentType'] not in (5120, 5121, 5122, 5123):
                raise ValueError('Invalid normalized component type')
            value = np.maximum(value.astype(np.float64)/np.iinfo(value.dtype).max, -1)
        if not np.isfinite(value).all(): raise ValueError('Nonfinite accessor values')
        return value

    def skin_weights(self, primitive, joint_count):
        attributes = primitive['attributes']; groups = []
        for name in attributes:
            if name.startswith(('JOINTS_', 'WEIGHTS_')):
                suffix = name.split('_', 1)[1]
                if not suffix.isdigit(): raise ValueError('Invalid skin attribute suffix')
                groups.append(int(suffix))
        groups = sorted(set(groups))
        if joint_count <= 0 or not groups or groups != list(range(len(groups))):
            raise ValueError('Contiguous skin attribute sets and positive joint count required')
        positions = self.read(attributes['POSITION'])
        dense = np.zeros((len(positions), joint_count))
        for group in groups:
            ji, wi = attributes.get(f'JOINTS_{group}'), attributes.get(f'WEIGHTS_{group}')
            if ji is None or wi is None: raise ValueError('Unpaired joint/weight attributes')
            ja, wa = self.document['accessors'][ji], self.document['accessors'][wi]
            if (ja['componentType'] not in (5121, 5123) or ja.get('normalized', False)
                    or wa['componentType'] not in (5126, 5121, 5123)
                    or (wa['componentType'] != 5126 and not wa.get('normalized', False))):
                raise ValueError('Invalid skin attribute component types')
            joints, weights = self.read(ji), self.read(wi)
            if (joints.shape != (len(positions), 4) or weights.shape != joints.shape
                    or (joints >= joint_count).any() or (weights < 0).any() or (weights > 1).any()):
                raise ValueError('Invalid skin attribute dimensions or values')
            np.add.at(dense, (np.arange(len(dense))[:, None], joints), weights)
        sums = dense.sum(axis=1)
        if (sums <= 0).any(): raise ValueError('Unweighted skin vertex')
        correction = float(np.max(np.abs(sums-1)))
        return dense/sums[:, None], correction
