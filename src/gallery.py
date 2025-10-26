import json, os
def _avg_vectors(vectors):
    if not vectors:
        return None
    from statistics import mean
    return [mean(vals) for vals in zip(*vectors)]


class Gallery:
    def __init__(self, path):
        self.path = path
        self.people = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                try:
                    self.people = json.load(f)
                except Exception:
                    self.people = {}
        else:
            self.people = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.people, f)

    def add_person(self, name, embedding):
        self.people.setdefault(name, []).append(embedding)
        self.save()


    def representative(self, name):
        """میانگین‌گیری ساده‌ی امبدینگ‌های یک فرد (نماینده)"""
        embs = self.people.get(name, [])
        return _avg_vectors(embs)
    
    def all(self):
        return self.people.items()
