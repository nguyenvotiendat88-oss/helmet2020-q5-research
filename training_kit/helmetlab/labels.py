"""Position order is explicit: D, P1, P2, P3, P0; 0 absent, 1 helmet, 2 no helmet."""
from __future__ import annotations
import numpy as np
from .common import read_json

POSITIONS = ("D", "P1", "P2", "P3", "P0")


class LabelMap:
    def __init__(self, path):
        self.data = read_json(path)
        self.classes = self.data["classes"]
        if self.data["positions"] != list(POSITIONS):
            raise ValueError("Position order must be D,P1,P2,P3,P0")
        if [c["id"] for c in self.classes] != list(range(len(self.classes))):
            raise ValueError("Class IDs must be contiguous and zero based")
        self.states = np.asarray([c["states"] for c in self.classes], dtype=np.int64)
        if self.states.shape != (len(self.classes), 5) or np.any((self.states < 0) | (self.states > 2)):
            raise ValueError("Each class must contain five states in {0,1,2}")
        if len(set(map(tuple, self.states))) != len(self.classes):
            raise ValueError("Two classes encode the same state")
        self.names = [c["name"] for c in self.classes]

    def __len__(self):
        return len(self.classes)

    def distribution(self, probabilities):
        """Normalize joint position likelihood over the known valid combinations."""
        p = np.asarray(probabilities, dtype=float)
        if p.shape[-2:] != (5, 3) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("Expected finite nonnegative [...,5,3] probabilities")
        if not np.allclose(p.sum(-1), 1, atol=1e-4):
            raise ValueError("Each position distribution must sum to one")
        ll = np.stack([np.log(np.maximum(p[..., np.arange(5), s], 1e-12)).sum(-1) for s in self.states], axis=-1)
        q = np.exp(ll - ll.max(-1, keepdims=True))
        return q / q.sum(-1, keepdims=True)

    def no_helmet_count(self, class_id):
        return int((self.states[class_id] == 2).sum())
