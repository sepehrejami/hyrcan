# src/attendance.py
import csv, os, time
from typing import Optional, Dict

class AttendanceStore:
    """
    ذخیره‌ی سوابق حضور در CSV:
      columns: timestamp_iso, date, name, score
    + مکانیسم cooldown روزانه: یک نفر در هر روز فقط یک‌بار ثبت می‌شود.
    """
    def __init__(self, csv_path="data/attendance.csv", daily_cooldown=True):
        self.csv_path = csv_path
        self.daily_cooldown = daily_cooldown
        self._seen_today: Dict[str, str] = {}  # name -> date "YYYY-MM-DD"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_iso", "date", "name", "score"])

    @staticmethod
    def _now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.localtime())

    def mark(self, name: str, score: Optional[float] = None) -> bool:
        """
        ثبت حضور. اگر قبلاً امروز برای این فرد ثبت‌شده باشد و daily_cooldown=True،
        چیزی نمی‌نویسد و False برمی‌گرداند. در غیر این صورت True.
        """
        today = self._today()
        if self.daily_cooldown and self._seen_today.get(name) == today:
            return False
        ts = self._now_iso()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts, today, name, f"{score:.4f}" if score is not None else ""])
        self._seen_today[name] = today
        return True

    def read_all(self):
        rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        return rows
