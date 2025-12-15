import numpy as np

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)

    ba = a - b
    bc = c - b

    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))
