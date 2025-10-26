
# Smart Attendance (MVP)

## ویژگی‌ها
- تشخیص و شناسایی چهره (Facenet پیش‌فرض، متریک Cosine/Euclidean)
- افزودن فرد جدید به گالری (JSON)
- پیام خوشامد شخصی‌سازی‌شده
- ثبت حضور روزانه در **CSV** + دکمهٔ دانلود گزارش در UI
- آستانهٔ شناسایی و متریک قابل تنظیم از UI

---

## پیش‌نیاز
- توصیهٔ نسخهٔ پایتون: **3.10 یا 3.11**  
  (DeepFace/TensorFlow ممکن است روی 3.13 ناپایدار باشد)
- وب‌کم و مجوز دسترسی مرورگر (برای Streamlit)

---

## نصب
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
````

> در اولین اجرا ممکن است مدل‌ها دانلود شوند (زمان‌بر است).

---

## اجرا

### 1) دمو‌ی وب (پیشنهادی)

```bash
streamlit run streamlit_app.py
```
به هیچ عنوان با ران کردن app.py پروژه را اجرا نکنید 
* در مرورگر: یک شات بگیر → اگر Unknown بود، نام بده و **Add** را بزن.
* ستون راست: جدول حضور و دکمهٔ **Download CSV**.
* تنظیمات سمت چپ: Threshold و Distance metric.



---

## پیکربندی

`config.yaml`:

```yaml
model_name: Facenet          # VGG-Face | Facenet | ArcFace | ...
distance_metric: cosine      # cosine | euclidean
recognition_threshold: 0.35  # هرچه کمتر، سخت‌گیرانه‌تر
min_face_size: 80            # فیلتر چهره‌های کوچک
gallery_path: "data/gallery.json"
```

---

## ساختار پوشه‌ها

```
smart-attendance/
├─ app.py                    # رانر CLI
├─ streamlit_app.py          # دمو UI
├─ requirements.txt
├─ config.yaml
├─ data/
│  ├─ gallery.json           # گالری افراد (امبدینگ‌ها)
│  └─ attendance.csv         # گزارش حضور
└─ src/
   ├─ face_service.py        # DeepFace detect+embed (+min_face_size)
   ├─ recognizer.py          # تطبیق امبدینگ‌ها با آستانه/متریک
   ├─ gallery.py             # مدیریت گالری JSON
   ├─ attendance.py          # ثبت حضور CSV با cooldown روزانه
   ├─ responder.py           # پیام خوشامد (LLM اختیاری)
   └─ utils.py
```

---

## نکات و رفع اشکال سریع

* **هیچ چهره‌ای پیدا نمی‌شود**: نور مناسب/نزدیکی به دوربین؛ `min_face_size` را کاهش دهید.
* **شناسایی اشتباهی/زیاد Unknown**: `recognition_threshold` را کمی بالا/پایین کنید (۰٫۳۰–۰٫۴۵).
* **Streamlit Connection error بعد از Add**: در نسخهٔ فعلی state مدیریت شده است؛ صفحه را یک‌بار Refresh کنید.
* **TensorFlow/DeepFace خطا**: از Python 3.10/3.11 استفاده کنید و در صورت نیاز:

  ```bash
  pip install tensorflow==2.15.*
  ```

---


## روش کار (خلاصه) 
تشخیص/امبدینگ: DeepFace (Facenet) با RetinaFace؛ فیلتر min_face_size برای نویز.

شناسایی: فاصله‌ی Cosine/Euclidean با آستانهٔ قابل‌تنظیم؛ نگه‌داری چند امبدینگ برای هر فرد.

افزودن فرد جدید: ذخیره‌ی امبدینگ در data/gallery.json.

حضور و غیاب: ثبت روزانه در data/attendance.csv و دانلود از UI.


---


## چرا من
من یک AI Engineer با تجربه‌ی عملی در Computer Vision و NLP هستم که پروژه‌های کاربردی را از ایده تا دمو و تحویل جلو برده‌ام. در این MVP، مسئله را به شکلی ماژولار و قابل‌گسترش حل کردم (تشخیص/امبدینگ با DeepFace، تطبیق با آستانه‌ی قابل‌تنظیم، گالری JSON، و ثبت حضور CSV همراه با Streamlit UI). ترکیب تجربه‌ام در طراحی پایپ‌لاین‌های بینایی ماشین و زمان‌-‌سری (تشخیص تغییرات، سوپررزولوشن، مدل‌های ترنسفورمری) با مهارت‌های Python/C++ باعث می‌شود بتوانم هم روی کیفیت مدل و هم روی مهندسی محصول تمرکز کنم. هدفم تحویل راه‌حل‌های ساده، تمیز و قابل‌نمایش است که سریع تست و توسعه می‌شوند.

---

---


