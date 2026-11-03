"""确定性的阈值连通分量；相似组不是身份。"""
import numpy as np


def face_components(rows, threshold=.5):
    """分块比较，边界包含阈值。排序键由调用者提供，结果不依赖输入顺序。"""
    rows = sorted(rows, key=lambda r: r['key'])
    if not rows:
        return []
    vectors = np.stack([np.asarray(r['vector'], dtype=np.float32) for r in rows])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms <= 0).any():
        raise ValueError('人脸向量无效')
    vectors = vectors / norms
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for start in range(0, len(rows), 256):
        scores = vectors[start:start + 256] @ vectors.T
        for local, j in np.argwhere(scores >= threshold):
            i = start + int(local)
            if j <= i:
                continue
            a, b = root(i), root(int(j))
            parents[max(a, b)] = min(a, b)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row['key'])
    return list(groups.values())
