from scipy.spatial.distance import cosine, euclidean

def _distance(a, b, metric="cosine"):
    if metric == "euclidean":
        return euclidean(a, b)
    return cosine(a, b)

class Recognizer:
    def __init__(self, gallery, threshold=0.35, metric="cosine"):
        self.gallery = gallery
        self.threshold = threshold
        self.metric = metric

    def match(self, embedding):
        best_name, best_score = None, 1e9
        for name, embs in self.gallery.all():
            if not embs:
                continue
            scores = [_distance(embedding, e, self.metric) for e in embs]
            score = min(scores) if scores else 1e9
            if score < best_score:
                best_score, best_name = score, name
        if best_score < self.threshold:
            return best_name, best_score
        return None, best_score
