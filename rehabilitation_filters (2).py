# rehabilitation_filters.py

"""
rehabilitation_filters.py
=========================

مكتبة الفلاتر النقية المستخدمة بواسطة RehabilitationEngineFinal.
كل دالة هنا:
- تأخذ PIL.Image (وأحيانًا weak_mask)
- ترجع PIL.Image محسنة
- لا تحتوي على logging داخلي، ولا تحميل نماذج، ولا prompt، ولا device
- تعتمد على PIL + numpy + cv2 فقط (أو أقل قدر ممكن)

الفلاتر مصممة لتكون pure functions → يسهل اختبارها وتوسيعها.
"""

from typing import Optional
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2


# ────────────────────────────────────────────────
#  الجزء 2 : دوال مساعدة صغيرة مشتركة
# ────────────────────────────────────────────────

def _ensure_rgb(img: Image.Image) -> Image.Image:
    """تحويل الصورة إلى RGB إذا لم تكن كذلك"""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _to_cv2_bgr(img: Image.Image) -> np.ndarray:
    """تحويل PIL إلى cv2 BGR"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _from_cv2_bgr(cv_img: np.ndarray) -> Image.Image:
    """تحويل cv2 BGR إلى PIL RGB"""
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ────────────────────────────────────────────────
#  الجزء 3 : basic_denoise_sharpen
# ────────────────────────────────────────────────
def basic_denoise_sharpen(
    img: Image.Image,
    denoise_level: int = 1,           # 0 = بدون، 1 = خفيف، 2 = متوسط
    sharpen_factor: float = 1.20,     # 1.0 = بدون تغيير، >1 = sharpen
    weak_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    فلتر أساسي: إزالة ضوضاء خفيفة + زيادة الحدة

    Parameters:
        denoise_level : مستوى الـ denoising (0–2)
        sharpen_factor: مقدار الـ sharpening (عادة 1.1 إلى 1.5)
        weak_mask     : إذا وُجد، يمكن تطبيق sharpen أقوى في المناطق الضعيفة

    Returns:
        صورة PIL محسنة
    """
    img = _ensure_rgb(img)

    # 1. Denoise (بسيط جدًا – median filter)
    if denoise_level >= 1:
        size = 3 if denoise_level == 1 else 5
        img = img.filter(ImageFilter.MedianFilter(size=size))

    # 2. Sharpen
    if sharpen_factor != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpen_factor)

    # إذا وجد weak_mask → sharpen إضافي موضعي (اختياري – بسيط)
    if weak_mask is not None and weak_mask.size == img.width * img.height:
        try:
            arr = np.array(img, dtype=np.float32)
            mask = (weak_mask > 127).astype(float)  # 0 أو 1

            # زيادة الحدة في المناطق الضعيفة فقط
            from scipy.ndimage import gaussian_filter
            base = gaussian_filter(arr, sigma=1.0)
            detail = arr - base
            enhanced_detail = detail * (1.0 + mask * 0.4)   # +40% في المناطق الضعيفة
            arr = np.clip(base + enhanced_detail, 0, 255).astype(np.uint8)

            img = Image.fromarray(arr)
        except Exception:
            # fallback صامت إذا scipy غير موجود
            pass

    return img


# ────────────────────────────────────────────────
#  الجزء 4 : advanced_contrast_color + color_balance
# ────────────────────────────────────────────────
def advanced_contrast_color(
    img: Image.Image,
    contrast_factor: float = 1.15,
    brightness_factor: float = 1.05,
    color_saturation: float = 1.10,
    weak_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    تحسين متقدم للتباين + السطوع + التشبع (saturation)

    Parameters:
        contrast_factor    : مقدار التباين (1.0 = بدون تغيير، >1 = أقوى)
        brightness_factor  : تعديل السطوع (1.0 = بدون تغيير)
        color_saturation   : تشبع الألوان (1.0 = بدون تغيير، >1 = ألوان أقوى)
        weak_mask          : إذا وُجد، يمكن تطبيق تحسين أقوى في المناطق الضعيفة

    Returns:
        صورة PIL بعد التحسين
    """
    img = _ensure_rgb(img)

    # 1. Brightness ثم Contrast
    if brightness_factor != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    if contrast_factor != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 2. Color saturation
    if color_saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(color_saturation)

    # إذا وجد weak_mask → تطبيق تحسين إضافي موضعي (اختياري)
    if weak_mask is not None and weak_mask.size == img.width * img.height:
        try:
            arr = np.array(img, dtype=np.float32)

            # mask = 1 في المناطق الضعيفة، 0 في باقي الصورة
            mask = (weak_mask > 127).astype(float)

            # زيادة التباين والتشبع محليًا
            mean = np.mean(arr, axis=(0,1), keepdims=True)
            enhanced = (arr - mean) * (1.0 + mask[..., None] * 0.25) + mean
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

            img = Image.fromarray(enhanced)
        except:
            # صامت إذا فشل (مثلاً numpy غير متوفر)
            pass

    return img


def color_balance(
    img: Image.Image,
    clip_limit: float = 2.0,
    weak_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    توازن ألوان تلقائي بسيط (مستوحى من auto levels / CLAHE-like)

    Parameters:
        clip_limit   : الحد الأقصى للـ contrast في CLAHE (إذا استخدمنا cv2)
        weak_mask    : غير مستخدم حاليًا في النسخة البسيطة

    Returns:
        صورة متوازنة الألوان
    """
    img = _ensure_rgb(img)

    try:
        # نسخة بسيطة باستخدام PIL فقط
        from PIL import ImageOps
        return ImageOps.autocontrast(img, cutoff=2)

    except:
        # fallback: زيادة contrast خفيفة إذا فشل autocontrast
        return ImageEnhance.Contrast(img).enhance(1.12)
    

# ────────────────────────────────────────────────
# Inpainting للـ artifacts
# ────────────────────────────────────────────────
def artifact_inpainting(
    img: Image.Image,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.75,
    model_name: str = "lama",            # "lama", "sd-inpaint", "mat", "powerpaint"
    device: str = "cpu",
) -> Image.Image:
    """
    إصلاح الـ artifacts (تشوهات، blur، noise، إلخ) باستخدام inpainting

    Parameters:
        mask     : قناع ثنائي (255 = المنطقة المراد إصلاحها)
                   إذا None → يُنشأ تلقائياً بناءً على weak areas
        strength : قوة التدخل (0.0 = بدون تغيير، 1.0 = إعادة رسم كامل)

    Returns:
        صورة PIL بعد الإصلاح (أو الأصلية إذا فشل)
    """
    img = _ensure_rgb(img)

    # إذا ما فيش ماسك → ننشئ واحد افتراضي (placeholder)
    if mask is None:
        # مثال بسيط: كل المناطق ذات الحدة المنخفضة
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        mask = (np.abs(lap) < 8).astype(np.uint8) * 255

    if mask.sum() < 100:  # لو الماسك صغير جداً → بدون تغيير
        return img

    try:
        # ────────────────────────────────
        #  محاولة LaMa (الأكثر توازناً حالياً)
        # ────────────────────────────────
        if model_name.lower() in ("lama", "big-lama"):
            from lama_cleaner.model import LaMa
            from lama_cleaner.schema import Config

            model = LaMa(device=device)
            config = Config(
                ldm_steps=25,
                ldm_sampler="plms",
                hd_strategy="original",
                hd_strategy_crop_margin=32,
                hd_strategy_crop_trigger_size=800,
                hd_strategy_resize_limit=2048,
            )

            result = model(
                image=np.array(img),
                mask=mask,
                config=config
            )
            return Image.fromarray(result)

    except ImportError:
        # lama_cleaner أو torch غير موجود
        pass
    except Exception:
        # فشل التحميل / GPU / weights / إلخ
        pass

    # ────────────────────────────────
    #  Fallback بسيط جداً (OpenCV Telea / Navier-Stokes)
    # ────────────────────────────────
    try:
        img_cv = _to_cv2_bgr(img)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        dst = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
        return _from_cv2_bgr(dst)
    except:
        pass

    # ────────────────────────────────
    #  Fallback أخير (blur خفيف على الماسك فقط)
    # ────────────────────────────────
    try:
        arr = np.array(img, dtype=np.float32)
        mask_float = mask.astype(float) / 255.0
        mask_3d = mask_float[..., None]

        blurred = cv2.GaussianBlur(arr, (21, 21), 0)
        result = arr * (1 - mask_3d) + blurred * mask_3d * strength
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    except:
        return img
    
    
# ────────────────────────────────────────────────
#  الجزء 5 : face_enhance
# ────────────────────────────────────────────────
def face_enhance(
    img: Image.Image,
    strength: float = 0.85,              # قوة التحسين العامة (0.0–1.0)
    face_confidence: float = 0.5,        # عتبة ثقة الكشف (أقل = يكشف وجوه أكثر)
    model_name: str = "codeformer",      # "codeformer", "gfpgan", "parsenet", "restoreformer"
    device: str = "cpu",                 # "cpu" أو "cuda" إذا متوفر
    upsample: int = 1,                   # 1 = بدون، 2 = ×2 (يحتاج ذاكرة أكبر)
    model_root: Optional[str] = None,    # مسار مخصص للـ weights (اختياري)
) -> Image.Image:
    """
    تحسين الوجوه باستخدام facexlib (يدعم CodeFormer / GFPGAN / RestoreFormer / ...)

    - إذا facexlib غير مثبت أو فشل التحميل → fallback بسيط (sharpen + contrast)
    - لا logging داخل الدالة (يترك للـ engine)
    - تعمل على PIL.Image فقط (لا تحتاج تحويل داخلي)

    Returns:
        PIL.Image محسنة (أو الأصلية إذا فشل التحسين)
    """
    img = _ensure_rgb(img)

    try:
        # استيراد facexlib (يجب أن يكون مثبت: pip install facexlib)
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        # إعداد المساعد
        face_helper = FaceRestoreHelper(
            img_size=512,
            face_size=512,
            model_rootpath=model_root or "./weights/facexlib",  # يمكن تغييره
            model_name=model_name,
            device=device,
            upscale=upsample,
        )

        # التحسين الفعلي
        restored_img = face_helper.enhance_face(
            img,
            strength=strength,
            face_confidence_threshold=face_confidence
        )

        # إذا نجح → نرجع النتيجة
        if restored_img is not None and isinstance(restored_img, Image.Image):
            return restored_img

    except ImportError:
        # facexlib غير مثبت → fallback
        pass
    except Exception:
        # أي خطأ آخر (GPU غير متوفر، weights مش موجودة، إلخ) → fallback
        pass

    # ────────────────────────────────
    #  Fallback بسيط جدًا (بدون مكتبات خارجية)
    # ────────────────────────────────
    enhancer_sharp = ImageEnhance.Sharpness(img)
    img = enhancer_sharp.enhance(1.0 + strength * 0.9)

    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(1.0 + strength * 0.25)

    enhancer_color = ImageEnhance.Color(img)
    img = enhancer_color.enhance(1.0 + strength * 0.15)

    return img


# ────────────────────────────────────────────────
#  الجزء 6 : background_enhance + text_enhance + symmetry_enhance
# ────────────────────────────────────────────────
def background_enhance(
    img: Image.Image,
    contrast_boost: float = 1.12,
    saturation_boost: float = 1.08,
    blur_radius_if_needed: float = 0.0,
    weak_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    تحسين الخلفية بشكل عام:
    - زيادة التباين والتشبع قليلاً
    - اختياري: blur خفيف جدًا إذا كانت الخلفية مزعجة

    ملاحظة: لا يوجد فصل حقيقي بين foreground/background هنا (placeholder بسيط)
    في المستقبل يمكن توسيعه بـ segmentation (مثل rembg أو SAM)
    """
    img = _ensure_rgb(img)

    # 1. تحسين التباين والتشبع (يؤثر على الخلفية أكثر عادةً)
    if contrast_boost != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_boost)

    if saturation_boost != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation_boost)

    # 2. blur خفيف جدًا (إذا طلب المستخدم)
    if blur_radius_if_needed > 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius_if_needed))

    # weak_mask غير مستخدم حاليًا في النسخة البسيطة
    # لو أردت: يمكن تطبيق blur أقوى خارج المناطق الضعيفة (لكن معقد بدون segmentation)

    return img


def text_enhance(
    img: Image.Image,
    sharpen_factor: float = 1.35,
    contrast_factor: float = 1.15,
    weak_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    تحسين النصوص/الكتابة في الصورة (logos، subtitles، watermarks، إلخ)

    - زيادة الحدة + التباين الموضعي
    - يعمل بشكل أفضل إذا كانت النصوص سوداء/بيضاء على خلفية
    """
    img = _ensure_rgb(img)

    # 1. Sharpen قوي نسبيًا (لإبراز الحواف)
    img = ImageEnhance.Sharpness(img).enhance(sharpen_factor)

    # 2. زيادة التباين (يساعد في فصل النصوص عن الخلفية)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # إذا وجد weak_mask → sharpen إضافي في المناطق الضعيفة
    if weak_mask is not None and weak_mask.size == img.width * img.height:
        try:
            arr = np.array(img, dtype=np.float32)
            mask = (weak_mask > 127).astype(float)[..., None]  # للقنوات الثلاث

            # Laplacian لاستخراج الحواف + تضخيمها في المناطق الضعيفة
            from scipy.ndimage import laplace
            edges = np.abs(laplace(arr.mean(axis=2)))
            enhanced = arr + edges[..., None] * mask * 0.6
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

            img = Image.fromarray(enhanced)
        except:
            # fallback صامت
            pass

    return img


def symmetry_enhance(
    img: Image.Image,
    axis: str = "vertical",             # "vertical", "horizontal", "both", "auto"
    strength: float = 0.3,              # قوة التصحيح (0.0 = بدون، 1.0 = قوي)
    weak_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    محاولة تحسين التناسق/التماثل في الصورة (placeholder بسيط جدًا)

    حاليًا: مجرد mirror خفيف أو averaging بين الجانبين
    في المستقبل يمكن توسيعه بـ optical flow أو deep symmetry models

    Returns:
        صورة مع تناسق محسن قليلاً (غالبًا تغيير طفيف)
    """
    img = _ensure_rgb(img)
    w, h = img.size

    if strength <= 0.05:
        return img

    try:
        arr = np.array(img, dtype=np.float32) / 255.0

        if axis in ("vertical", "both", "auto"):
            # mirror الجانب الأيسر على الأيمن (أو العكس) بنسبة strength
            left = arr[:, :w//2, :]
            right = arr[:, w//2:, :]
            if left.shape[1] != right.shape[1]:
                right = right[:, ::-1, :]  # flip إذا الأحجام مختلفة
            blended = (1 - strength) * right + strength * left
            arr[:, w//2:, :] = blended[:, :right.shape[1], :]

        if axis in ("horizontal", "both"):
            # نفس الشيء رأسيًا
            top = arr[:h//2, :, :]
            bottom = arr[h//2:, :, :]
            blended = (1 - strength) * bottom + strength * top
            arr[h//2:, :, :] = blended[:bottom.shape[0], :, :]

        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    except:
        # fallback: بدون تغيير
        pass

    return img


# ────────────────────────────────────────────────
#  الجزء 7 : compute_clip_score + قائمة التصدير (__all__)npainting
# ────────────────────────────────────────────────
def compute_clip_score(
    img: Image.Image,
    prompt: str,
    clip_model=None,
    clip_processor=None,
    normalize: bool = True,
) -> float:
    """
    حساب درجة التوافق (CLIP score) بين الصورة والـ prompt

    Parameters:
        img             : الصورة (PIL Image)
        prompt          : النص الوصفي
        clip_model      : نموذج CLIP المحمل مسبقاً (من الـ engine)
        clip_processor  : المعالج المحمل مسبقاً
        normalize       : هل نرجع النتيجة بعد softmax (True = 0..1)

    Returns:
        float بين 0 و 1 (أعلى = توافق أفضل)
        أو 0.0 إذا فشل الحساب
    """
    if clip_model is None or clip_processor is None:
        return 0.0

    try:
        # تحضير الإدخال
        inputs = clip_processor(
            text=[prompt],
            images=[img],
            return_tensors="pt",
            padding=True
        )

        # حساب الـ logits
        with torch.no_grad():
            outputs = clip_model(**inputs)

        # استخراج الدرجة
        logits_per_image = outputs.logits_per_image
        if normalize:
            probs = logits_per_image.softmax(dim=1)
            score = probs[0][0].item()
        else:
            score = logits_per_image[0][0].item()

        return float(score)

    except Exception:
        # أي فشل (GPU، تحجيم، إلخ) → fallback صامت
        return 0.0


# ────────────────────────────────────────────────
#  قائمة التصدير (__all__) – لتحديد ما يُستورد عند from ... import *
# ────────────────────────────────────────────────

__all__ = [
    # الفلاتر الأساسية
    "basic_denoise_sharpen",
    "advanced_contrast_color",
    "color_balance",
    
    # تحسينات متخصصة
    "face_enhance",
    "background_enhance",
    "text_enhance",
    "symmetry_enhance",
    "artifact_inpainting",
    
    # مساعدات
    "compute_clip_score",
    
    # الدوال الداخلية المفيدة (اختياري)
    "_ensure_rgb",
    "_to_cv2_bgr",
    "_from_cv2_bgr",
]
    
    