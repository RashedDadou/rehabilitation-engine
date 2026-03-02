# rehabilitation_engine_final.py

"""
rehabilitation_engine_final.py
=============================

محرك إعادة تأهيل تلقائي للصور المولدة بالذكاء الاصطناعي
يهدف إلى تحسين الجودة، تقليل الـ artifacts، تعزيز التفاصيل، وتحسين التوافق مع الـ prompt.

المميزات الرئيسية:
- كشف تلقائي للمناطق الضعيفة (weak areas)
- تكرارات ديناميكية بناءً على عدد الوجوه + نسبة الضعف + CLIP score
- دعم أنماط متعددة: full / light / fast
- تحسين تطوري موحد (evolutionary_enhance)
- تقارير نصية + PDF + مقارنات بصرية

الاعتماديات الأساسية:
- PIL, numpy, opencv-python
- torch + transformers (اختياري لـ CLIP)
- reportlab (اختياري لـ PDF)

الفلاتر الفعلية تأتي من ملف منفصل: rehabilitation_filters.py
"""

import os
from datetime import datetime
import logging
from typing import List, Optional, Dict, Callable, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2

from rehabilitation_filters import *

# مكتبات AI اختيارية (مع fallback)
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


# استيراد الفلاتر (يجب أن يكون rehabilitation_filters.py في نفس المجلد)
try:
    from rehabilitation_filters import (
        basic_denoise_sharpen,
        advanced_contrast_color,
        genetic_mutation_selection,  # أو استبدلها بـ evolutionary_enhance إذا أردت
        artifact_inpainting,
        compute_clip_score,
        face_enhance,
        background_enhance,
        color_balance,
        text_enhance,
        symmetry_enhance,
    )
except ImportError:
    logger.warning("تعذر استيراد rehabilitation_filters → بعض المراحل ستكون معطلة")

# ثم داخل __init__ بعد تعريف self.filters:
self.filters.update({
    "basic": basic_denoise_sharpen,
    "advanced": advanced_contrast_color,
    "genetic": genetic_mutation_selection,   # أو يمكن تركه None واستخدام evolutionary_enhance دائمًا
    "inpainting": artifact_inpainting,
    "face_enhance": face_enhance,
    "background_enhance": background_enhance,
    "color_balance": color_balance,
    "text_enhance": text_enhance,
    "symmetry_enhance": symmetry_enhance,
    "clip_score": compute_clip_score,
})

levels = [
    "basic",
    "face_enhance",
    "color_balance",
    "advanced",
    "background_enhance",
    "text_enhance",
    "symmetry_enhance",
    "inpainting",
    "dna_revival"   # ← هنا بدل "genetic" أو أضفها معها
]

# logging إعداد أساسي
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class RehabilitationEngineFinal:
    """
    المحرك الرئيسي لإعادة تأهيل الصور المولدة.
    يدير الـ pipeline، يكشف المناطق الضعيفة، يحسب التكرارات ديناميكيًا،
    ويستدعي الفلاتر من ملف rehabilitation_filters.
    """

    def __init__(
        self,
        prompt: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() and torch.cuda.is_available() else "cpu",
        use_clip: bool = True,
        use_inpainting: bool = True,
        use_perceptual: bool = True,
        iteration_count: int = 4,
        mutation_rate: float = 0.015,
        var_threshold: float = 9.0,
        clip_threshold: float = 0.30,
        var_improvement: float = 1.03,
        face_confidence: float = 0.60,
        report_dir: str = "rehab_reports",
    ):
        self.prompt = prompt
        self.device = device

        # حالة توفر CLIP
        self.use_clip = use_clip and CLIP_AVAILABLE
        if use_clip and not CLIP_AVAILABLE:
            logger.warning("CLIP غير متوفر → تم تعطيل use_clip")

        self.use_inpainting = use_inpainting
        self.use_perceptual = use_perceptual

        # معاملات التحكم
        self.iteration_count = iteration_count
        self.mutation_rate = mutation_rate
        self.var_threshold = var_threshold
        self.clip_threshold = clip_threshold
        self.var_improvement = var_improvement
        self.face_confidence = face_confidence

        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

        # إحصائيات قبل وبعد
        self.initial_stats: Dict[str, float] = {}
        self.final_stats: Dict[str, float] = {}

        # تاريخ التقارير (اختياري)
        self.report_history: List[Dict] = []

        # ───────────────────────────────────────
        #  قاموس الفلاتر (الأسماء فقط – التنفيذ في rehabilitation_filters)
        # ───────────────────────────────────────
        self.filters: Dict[str, Callable] = {
            "basic":              None,  # basic_denoise_sharpen
            "advanced":           None,  # advanced_contrast_color
            "genetic":            None,  # genetic_mutation_selection أو ما يحل محله
            "inpainting":         None,  # artifact_inpainting
            "face_enhance":       None,  # face_enhance
            "background_enhance": None,  # background_enhance
            "color_balance":      None,  # color_balance
            "text_enhance":       None,  # text_enhance
            "symmetry_enhance":   None,  # symmetry_enhance
            "clip_score":         None,  # compute_clip_score
        }

        # سنقوم لاحقًا بملء القاموس فعليًا عند استيراد rehabilitation_filters
        # مثال: من rehabilitation_filters import * ثم self.filters["basic"] = basic_denoise_sharpen
        # لكن حاليًا نتركه None لنعرف الأسماء المتوقعة فقط

        # تحميل CLIP إذا كان مفعلاً
        self.clip_model = None
        self.clip_processor = None
        if self.use_clip:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                if self.device.startswith("cuda"):
                    self.clip_model = self.clip_model.to(self.device)
                logger.info("تم تحميل نموذج CLIP بنجاح")
            except Exception as e:
                logger.error(f"فشل تحميل CLIP: {e}")
                self.use_clip = False

        self.revival_boost_factor = 0.92   # قوة الإسقاط الأفقي (تعويض السالب)
        self.negative_exponent = 2.0       # exponent لتركيز الاحتمالية على السالب القوي


    # ────────────────────────────────────────────────
    #          الدوال التشخيصية والكشف (Diagnostics)
    # ────────────────────────────────────────────────
    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """
        تحضير الصورة: نسخ + تحويل إلى RGB
        (دالة بسيطة لكنها تُستخدم في كل مكان)
        """
        if image is None:
            raise ValueError("الصورة فارغة أو None")
        return image.copy().convert("RGB")


    def _detect_weak_areas(self, img: Image.Image) -> np.ndarray:
        """
        كشف المناطق الضعيفة (منخفضة الحدة / blur محتمل) باستخدام Laplacian variance
        ترجع mask ثنائي (0 أو 255)
        """
        if img is None or img.size[0] < 16 or img.size[1] < 16:
            logger.warning("صورة صغيرة جدًا → لا يمكن كشف مناطق ضعيفة")
            return np.zeros((8, 8), dtype=np.uint8)

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        abs_lap = np.abs(lap)

        # المناطق التي Laplacian variance منخفضة = مناطق ضعيفة
        threshold = self.var_threshold
        mask = (abs_lap < threshold).astype(np.uint8) * 255

        # توسيع بسيط للمناطق الضعيفة (اختياري، يساعد في تغطية الحواف)
        if mask.sum() > 0:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        weak_ratio = (mask.sum() / mask.size) * 100
        logger.debug(f"نسبة المناطق الضعيفة المكتشفة: {weak_ratio:.2f}%")

        return mask


    def _count_faces(self, img_array: np.ndarray) -> int:
        """
        محاولة عد الوجوه بأخف طريقة ممكنة
        - يفضل استخدام face_recognition إذا موجود
        - fallback بسيط جدًا إذا غير موجود
        """
        try:
            import face_recognition
            locations = face_recognition.face_locations(img_array)
            return len(locations)
        except ImportError:
            logger.debug("face_recognition غير مثبت → عد الوجوه = 0")
            return 0
        except Exception as e:
            logger.debug(f"خطأ في كشف الوجوه: {str(e)}")
            return 0


    def _extract_diagnostics(self, img: Image.Image) -> Dict[str, float]:
        """
        استخراج جميع المقاييس التشخيصية المهمة مرة واحدة
        ترجع قاموسًا موحدًا يُستخدم في كل مكان
        """
        if img is None:
            return {"valid": False, "error": "صورة فارغة"}

        array = np.array(img)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

        diagnostics = {
            "valid": True,
            "width": img.width,
            "height": img.height,
            "variance": float(np.var(array)),
            "sharpness": float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F)))),
            "weak_mask": self._detect_weak_areas(img),
        }

        # حساب نسبة المناطق الضعيفة
        weak_mask = diagnostics["weak_mask"]
        diagnostics["weak_areas_percentage"] = \
            (weak_mask.sum() / weak_mask.size) * 100 if weak_mask.size > 0 else 0.0

        # عدد الوجوه
        diagnostics["faces_detected"] = self._count_faces(array)

        # CLIP score (إذا مفعل ومتوفر)
        diagnostics["clip_score"] = 0.0
        if self.use_clip and self.clip_processor and self.clip_model and self.prompt:
            try:
                inputs = self.clip_processor(
                    text=[self.prompt],
                    images=[img],
                    return_tensors="pt",
                    padding=True
                )
                if self.device.startswith("cuda"):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                score = outputs.logits_per_image.softmax(dim=1)[0][0].item()
                diagnostics["clip_score"] = float(score)
            except Exception as e:
                logger.debug(f"فشل حساب CLIP score: {str(e)}")

        return diagnostics
    
    # ────────────────────────────────────────────────
    #     حساب عدد التكرارات الديناميكي (Dynamic Iterations)
    # ────────────────────────────────────────────────

    def calculate_dynamic_iterations(
        self,
        diagnostics: Dict[str, float],
        base_iterations: int = 3,
        max_iterations: int = 12,
        min_iterations: int = 2
    ) -> int:
        """
        حساب عدد التكرارات الديناميكي بناءً على حالة الصورة

        Parameters:
            diagnostics: الناتج من _extract_diagnostics
            base_iterations: عدد التكرارات الأساسي (افتراضي 3)
            max_iterations / min_iterations: الحدود

        Returns:
            عدد التكرارات المقترح (int)
        """
        if not diagnostics.get("valid", False):
            logger.warning("تشخيص غير صالح → استخدام العدد الأساسي")
            return base_iterations

        # استخراج القيم المهمة (مع قيم افتراضية آمنة)
        weak_pct     = diagnostics.get("weak_areas_percentage", 0.0)
        faces        = diagnostics.get("faces_detected", 0)
        clip_score   = diagnostics.get("clip_score", 0.0)
        sharpness    = diagnostics.get("sharpness", 0.0)
        variance     = diagnostics.get("variance", 0.0)

        # ────────────────────────────────
        #  عوامل التأثير (factors) – يمكن تعديلها بسهولة
        # ────────────────────────────────

        # 1. الوجوه (أهم عامل عادةً)
        face_factor = 1.0
        if faces >= 4:
            face_factor = 2.0       # وجوه كثيرة → جهد أكبر
        elif faces >= 2:
            face_factor = 1.6
        elif faces == 1:
            face_factor = 1.3
        elif faces == 0:
            face_factor = 0.8       # بدون وجوه → أقل تكرار

        # 2. المناطق الضعيفة
        weak_factor = 1.0
        if weak_pct > 45:
            weak_factor = 1.9
        elif weak_pct > 25:
            weak_factor = 1.5
        elif weak_pct > 10:
            weak_factor = 1.2
        elif weak_pct < 5:
            weak_factor = 0.7

        # 3. توافق CLIP (إذا موجود)
        clip_factor = 1.0
        if self.use_clip:
            if clip_score < 0.22:
                clip_factor = 2.1       # توافق ضعيف جدًا → نحتاج جهد كبير
            elif clip_score < 0.35:
                clip_factor = 1.6
            elif clip_score < 0.50:
                clip_factor = 1.25
            elif clip_score > 0.78:
                clip_factor = 0.65      # توافق عالي → يمكن تقليل التكرارات

        # 4. الحدة / الـ variance (عامل مساعد)
        quality_factor = 1.0
        if sharpness < 8.0 or variance < 400:
            quality_factor = 1.4 + (8.0 - sharpness) * 0.08
            quality_factor = min(quality_factor, 2.2)

        # ────────────────────────────────
        #  الحساب النهائي
        # ────────────────────────────────

        combined_factor = (
            face_factor *
            weak_factor *
            clip_factor *
            quality_factor
        )

        # عدد التكرارات الأولي
        dynamic = base_iterations * combined_factor

        # تقريب + حدود
        dynamic = round(dynamic)
        dynamic = max(min_iterations, min(dynamic, max_iterations))

        # تسجيل للـ debugging
        logger.info(
            f"حساب التكرارات الديناميكي → {dynamic} تكرارات\n"
            f"   face_factor={face_factor:.2f}   weak={weak_pct:.1f}%   "
            f"clip={clip_score:.3f}   quality_factor={quality_factor:.2f}"
        )

        return dynamic
    
    # ────────────────────────────────────────────────
    #          التحسين التطوري الموحد (Evolutionary Enhance)
    # ────────────────────────────────────────────────
    def evolutionary_enhance(
        self,
        img: Image.Image,
        iterations: int = 5,
        mutation_rate: float = 0.015,
        guidance: str = "variance+sharpness",
        clip_score_fn: Optional[Callable[[Image.Image, str], float]] = None,
        prompt: Optional[str] = None,
        weak_mask: Optional[np.ndarray] = None,
        min_improvement_ratio: float = 1.025,
        lpips_threshold: float = 0.018,
        max_no_improvement: int = 4,
        sharpness_boost_range: Tuple[float, float] = (1.05, 1.22),
        contrast_boost_range: Tuple[float, float] = (1.00, 1.18),
    ) -> Image.Image:
        """
        تحسين تطوري موحد يدعم أنماط توجيه متعددة:
          - "variance+sharpness"   → مزيج variance + حدة (افتراضي)
          - "clip"                 → يعتمد على clip_score_fn
          - "lpips"                → يحتاج lpips_fn (غير مدعوم حالياً في هذا النسخة)
          - "basic"                → تحسينات خفيفة فقط بدون طفرة قوية

        لا تعرف فلاتر داخلها – تعتمد على clip_score_fn فقط إذا مرر.
        """
        if guidance not in {"variance+sharpness", "clip", "basic"}:
            logger.warning(f"guidance غير مدعوم: {guidance} → استخدام 'variance+sharpness'")
            guidance = "variance+sharpness"

        current = img.copy().convert("RGB")
        best = current.copy()

        best_score = self._fitness_score(
            best, guidance=guidance, clip_score_fn=clip_score_fn, prompt=prompt
        )

        no_improve_count = 0
        logger.info(f"بدء evolutionary_enhance | guidance={guidance} | baseline_score={best_score:.4f}")

        for i in range(iterations):
            candidate = self._mutate_and_enhance(
                current,
                mutation_rate=mutation_rate,
                weak_mask=weak_mask,
                sharpness_boost_range=sharpness_boost_range,
                contrast_boost_range=contrast_boost_range,
            )

            candidate_score = self._fitness_score(
                candidate, guidance=guidance, clip_score_fn=clip_score_fn, prompt=prompt
            )

            improved = self._is_better(
                candidate_score, best_score, guidance,
                min_improvement_ratio, lpips_threshold
            )

            if improved:
                logger.debug(f"Iter {i+1:2d} تحسن → {candidate_score:.4f} (Δ {candidate_score - best_score:+.4f})")
                best = candidate.copy()
                best_score = candidate_score
                current = candidate
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= max_no_improvement:
                    logger.info(f"توقف مبكر بعد {i+1} تكرارات (لا تحسن ملحوظ)")
                    break

        logger.info(f"انتهى evolutionary_enhance بعد {i+1} تكرارات | أفضل score = {best_score:.4f}")
        return best


    def _mutate_and_enhance(
        self,
        img: Image.Image,
        mutation_rate: float,
        weak_mask: Optional[np.ndarray],
        sharpness_boost_range: Tuple[float, float],
        contrast_boost_range: Tuple[float, float],
    ) -> Image.Image:
        """إنشاء مرشح (mutant) بطفرة + تحسينات خفيفة"""
        mutated = img.copy()

        # 1. تحسينات deterministic عشوائية خفيفة
        s_boost = np.random.uniform(*sharpness_boost_range)
        c_boost = np.random.uniform(*contrast_boost_range)

        if abs(s_boost - 1.0) > 0.005:
            mutated = ImageEnhance.Sharpness(mutated).enhance(s_boost)
        if abs(c_boost - 1.0) > 0.005:
            mutated = ImageEnhance.Contrast(mutated).enhance(c_boost)

        # 2. طفرة بكسلية (أقوى في المناطق الضعيفة إن وجدت)
        if mutation_rate > 1e-6:
            arr = np.array(mutated, dtype=np.float32)
            h, w, _ = arr.shape

            if weak_mask is not None and weak_mask.size == h * w:
                prob = np.clip(weak_mask.astype(float) / 255 * 2.5 + 0.12, 0, 1)
                change_mask = np.random.rand(h, w) < prob
            else:
                change_mask = np.random.rand(h, w) < mutation_rate

            if change_mask.any():
                noise = np.random.normal(0, 10.5, size=(change_mask.sum(), 3))
                arr[change_mask] += noise
                arr = np.clip(arr, 0, 255)

            mutated = Image.fromarray(arr.astype(np.uint8))

        # 3. فلتر خفيف عشوائي (ليس دائمًا)
        if np.random.rand() < 0.65:
            mutated = mutated.filter(ImageFilter.MedianFilter(size=3))

        return mutated


    def _fitness_score(
        self,
        img: Image.Image,
        guidance: str,
        clip_score_fn: Optional[Callable],
        prompt: Optional[str],
    ) -> float:
        """حساب درجة الجودة حسب نمط التوجيه"""
        array = np.array(img)

        if guidance == "clip" and clip_score_fn and prompt:
            try:
                score = clip_score_fn(img, prompt)
                return float(score)  # أعلى أفضل
            except Exception as e:
                logger.debug(f"CLIP فشل في fitness: {str(e)}")
                # fallback إلى variance+sharpness
                guidance = "variance+sharpness"

        if guidance in ("variance+sharpness", "basic"):
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            var = float(np.var(array))
            sharp = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))
            return var * 0.35 + sharp * 1.0

        # fallback آمن جدًا
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        return float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))


    def _is_better(
        self,
        new_score: float,
        old_score: float,
        guidance: str,
        min_ratio: float,
        lpips_delta: float,
    ) -> bool:
        """هل النتيجة الجديدة أفضل؟"""
        if guidance == "lpips":
            # lpips: أقل أفضل → نحتاج فرق سالب
            return new_score < old_score - lpips_delta

        # باقي الأنماط: أعلى أفضل
        return new_score > old_score * min_ratio
    
    # ────────────────────────────────────────────────
    #          تطبيق مرحلة واحدة من الـ pipeline
    # ────────────────────────────────────────────────
    def dna_pulsed_revival_with_refine(
        self,
        img: Image.Image,
        prompt: Optional[str] = "high quality realistic image, detailed, sharp, no artifacts",
        base_mutation_intensity: float = 0.18,
        max_pulses: int = 4,
        revival_boost_factor: float = 0.85,
        negative_exponent: float = 2.0,
        use_lpips_check: bool = True,
        ref_img: Optional[Image.Image] = None,
        refine_steps_total: int = 28,           # إجمالي خطوات الـ refine
        initial_refine_strength: float = 0.72,
        min_lpips_stop: float = 0.22,           # إذا وصل أقل من كده → توقف الـ refine
    ) -> Image.Image:
        """
        إحياء جيني نبضي + Refine نبضي مدمج
        1. طفرة أولية مركزة على المناطق الميتة
        2. نبضات إحياء متدرجة (revival pulses)
        3. نبضات Refine (denoising جزئي) مع فحص LPIPS بين النبضات
        4. تلميع نهائي خفيف
        """
        logger.info("بدء dna_pulsed_revival_with_refine ...")

        img = img.convert("RGB")
        current = np.array(img, dtype=np.float32)
        h, w, _ = current.shape

        # ─── 0. إنشاء ماسك المناطق الضعيفة / الميتة ───
        weak_mask = self._auto_weak_mask(current)           # افتراض أن هذه موجودة في الكلاس
        negative_strength = weak_mask.astype(float) / 255.0

        dead_mask = negative_strength > 0.38                # عتبة "ميتة"
        dead_ratio = np.mean(dead_mask)
        logger.info(f"نسبة المناطق الميتة: {dead_ratio:.1%}")

        # ─── 1. طفرة أولية مركزة فقط على المناطق الميتة ───
        if dead_mask.any():
            noise_scale = np.where(dead_mask, base_mutation_intensity * 52, 0.0)
            noise = np.random.normal(0, noise_scale[..., None], size=(h, w, 3))
            current[dead_mask] += noise[dead_mask]
            current = np.clip(current, 0, 255)

        current_img = Image.fromarray(current.astype(np.uint8))

        # ─── 2. حساب عدد النبضات (مشترك للإحياء والـ refine) ───
        num_pulses = min(max_pulses, 1 + int(dead_ratio * 9))   # 1 إلى 4 نبضات
        logger.info(f"عدد النبضات المخطط: {num_pulses}")

        # ─── 3. النبضات الإحيائية (revival phase) ───
        for pulse in range(1, num_pulses + 1):
            pulse_factor = 0.58 + (pulse - 1) * 0.24          # 0.58 → 0.82 → 1.06 → 1.30

            # تعزيز من الجيران (revival boost)
            kernel_size = 9 + pulse * 2
            neighbors = cv2.GaussianBlur(current, (kernel_size, kernel_size), 0)
            revival_boost = (neighbors - current) * revival_boost_factor * pulse_factor * negative_strength[..., None]

            current += revival_boost
            current = np.clip(current, 0, 255)

            # sharpen خفيف بعد كل نبضة
            current_img = Image.fromarray(current.astype(np.uint8))
            current_img = current_img.filter(ImageFilter.SHARPEN)
            current = np.array(current_img, dtype=np.float32)

        # ─── 4. مرحلة الـ Refine النبضي (إذا كان الـ pipeline موجود) ───
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            refine_steps_per_pulse = refine_steps_total // num_pulses
            current_strength = initial_refine_strength

            for pulse in range(num_pulses):
                mask_img = Image.fromarray(weak_mask)

                try:
                    refined = self.pipeline(
                        prompt=prompt,
                        image=current_img,
                        mask_image=mask_img,
                        strength=current_strength,
                        num_inference_steps=refine_steps_per_pulse,
                        guidance_scale=7.2,
                    ).images[0]
                except Exception as e:
                    logger.warning(f"فشل خطوة Refine نبضة {pulse+1}: {e}")
                    break

                # فحص LPIPS → توقف مبكر إذا جودة كافية
                if use_lpips_check and hasattr(self, 'lpips_loss') and self.lpips_loss and ref_img:
                    dist = self.lpips_loss(
                        np.array(ref_img).transpose(2,0,1)[None],
                        np.array(refined).transpose(2,0,1)[None]
                    ).item()

                    logger.info(f"نبضة Refine {pulse+1} → LPIPS = {dist:.4f}")
                    if dist < min_lpips_stop:
                        logger.info(f"توقف مبكر عند LPIPS {dist:.4f}")
                        break

                current_img = refined
                current_strength *= 0.82   # تقليل تدريجي للقوة

        # ─── 5. تلميع نهائي خفيف (post-processing) ───
        current_img = current_img.filter(ImageFilter.SHARPEN)
        current_img = ImageEnhance.Contrast(current_img).enhance(1.10)
        current_img = ImageEnhance.Color(current_img).enhance(1.07)
        current_img = ImageEnhance.Sharpness(current_img).enhance(1.15)

        logger.info("انتهى dna_pulsed_revival_with_refine")
        return current_img


    
    def _apply_stage(
        self,
        img: Image.Image,
        stage: str,
        weak_mask: Optional[np.ndarray] = None,
        diagnostics: Optional[Dict] = None,
    ) -> Image.Image:
        """
        تطبيق مرحلة (stage) واحدة من قائمة levels
        
        Parameters:
            img: الصورة الحالية
            stage: اسم المرحلة (مثل "basic", "face_enhance", "genetic", ...)
            weak_mask: قناع المناطق الضعيفة (يُمرر للمراحل التي تحتاجه)
            diagnostics: نتائج التشخيص (يُستخدم أحيانًا لاتخاذ قرارات)

        Returns:
            الصورة بعد تطبيق المرحلة (أو الصورة الأصلية إذا فشل التطبيق)
        """
        
        if stage not in self.filters or self.filters[stage] is None:
            logger.warning(f"المرحلة '{stage}' غير معرفة أو غير محملة → تم تخطيها")
            return img

        fn = self.filters[stage]


        # ────────────────────────────────
        #  حالات خاصة لبعض المراحل
        # ────────────────────────────────

        if stage == "genetic":
            # نستخدم الدالة الموحدة evolutionary_enhance
            iters = self.iteration_count
            if diagnostics:
                # إذا مررنا diagnostics → نحسب عدد التكرارات ديناميكيًا
                try:
                    iters = self.calculate_dynamic_iterations(diagnostics)
                except Exception as e:
                    logger.debug(f"فشل حساب التكرارات الديناميكية: {e} → استخدام القيمة الافتراضية")

            logger.info(f"تطبيق المرحلة genetic → {iters} تكرارات")
            
            return self.evolutionary_enhance(
                img=img,
                iterations=iters,
                mutation_rate=self.mutation_rate,
                guidance="clip" if self.use_clip and self.prompt else "variance+sharpness",
                clip_score_fn=self.filters.get("clip_score"),
                prompt=self.prompt,
                weak_mask=weak_mask,
            )

        # ────────────────────────────────
        #  المراحل العادية (الفلاتر من rehabilitation_filters)
        # ────────────────────────────────

        kwargs = {}

        # تمرير weak_mask للمراحل التي تحتاجه
        if weak_mask is not None and stage in {
            "inpainting",
            "color_balance",
            "text_enhance",
            "symmetry_enhance",
            # أضف هنا أي مراحل أخرى تحتاج weak_mask في المستقبل
        }:
            kwargs["weak_mask"] = weak_mask

        # معاملات خاصة ببعض المراحل
        if stage == "face_enhance":
            kwargs["confidence"] = self.face_confidence

        # محاولة تطبيق الفلتر
        try:
            logger.debug(f"تطبيق المرحلة: {stage}")
            result = fn(img, **kwargs)
            
            # التحقق من أن النتيجة صورة PIL صالحة
            if not isinstance(result, Image.Image):
                logger.warning(f"المرحلة {stage} لم ترجع صورة PIL → إرجاع الصورة الأصلية")
                return img
                
            return result

        except Exception as e:
            logger.error(f"خطأ أثناء تطبيق المرحلة '{stage}': {str(e)}")
            # fallback: نرجع الصورة بدون تغيير
            return img
        
        if stage == "dna_revival":
            return self.dna_inspired_single_revival(
                img,
                mutation_intensity=self.mutation_rate,
                weak_mask=weak_mask,   # يمكن أن يكون None → auto
                use_lpips_guidance=True,
                ref_img=image          # الصورة الأصلية
            )
    
    
    # ────────────────────────────────────────────────
    #              الدالة الرئيسية: rehabilitate
    # ────────────────────────────────────────────────
    def rehabilitate(
        self,
        image: Image.Image,
        levels: Optional[List[str]] = None,
        mode: str = "full",
        fast_iterations: int = 3,
        light_strength: float = 0.7,
        return_stats: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, Dict]]:
        """
        الدالة الرئيسية لإعادة تأهيل الصورة

        Parameters:
            image          : الصورة المدخلة (PIL Image)
            levels         : قائمة المراحل المطلوب تطبيقها (إذا None → الافتراضية)
            mode           : "full" | "light" | "fast"
            fast_iterations: عدد التكرارات في الوضع light/fast إذا استخدمنا evolutionary
            light_strength : قوة التحسين في الوضع light (0.0–1.0)
            return_stats   : إذا True → يرجع (الصورة، dict الإحصائيات النهائية)

        Returns:
            الصورة المحسنة، أو (الصورة، dict) إذا return_stats=True
        """
        if image is None or image.width < 8 or image.height < 8:
            raise ValueError("الصورة غير صالحة أو صغيرة جدًا")

        current = self._prepare_image(image)

        # القائمة الافتراضية للوضع full
        if levels is None:
            levels = [
                "basic",
                "face_enhance",
                "color_balance",
                "advanced",
                "background_enhance",
                "text_enhance",
                "symmetry_enhance",
                "inpainting",
                "genetic"   # أو evolutionary_enhance مباشرة
            ]

        # ────────────────────────────────
        #          وضع fast (أسرع ما يمكن)
        # ────────────────────────────────
        if mode == "fast":
            if "basic" in self.filters and self.filters["basic"]:
                current = self.filters["basic"](current)
            current = current.filter(ImageFilter.SHARPEN)
            final_stats = self._extract_diagnostics(current)
            if return_stats:
                return current, final_stats
            return current

        # ────────────────────────────────
        #         وضع light (متوازن سريع)
        # ────────────────────────────────
        if mode == "light":
            # 1. تحسين الوجوه إذا موجود
            if "face_enhance" in self.filters and self.filters["face_enhance"]:
                current = self.filters["face_enhance"](
                    current,
                    strength=light_strength,
                    face_confidence=self.face_confidence
                )

            # 2. تحسين تطوري خفيف
            current = self.evolutionary_enhance(
                current,
                iterations=fast_iterations,
                mutation_rate=self.mutation_rate * 0.6,
                guidance="variance+sharpness",
                weak_mask=None  # بدون mask في الوضع الخفيف عادة
            )

            # 3. sharpen نهائي
            current = current.filter(ImageFilter.SHARPEN)

            final_stats = self._extract_diagnostics(current)
            if return_stats:
                return current, final_stats
            return current

        # ────────────────────────────────
        #          الوضع full (الكامل)
        # ────────────────────────────────

        # 1. التشخيص الأولي + استخراج weak_mask
        initial_diag = self._extract_diagnostics(current)
        self.initial_stats = initial_diag.copy()
        weak_mask = initial_diag.get("weak_mask")

        # 2. Inpainting مبكر إذا كانت المناطق الضعيفة كثيرة
        if self.use_inpainting and "inpainting" in levels:
            if initial_diag.get("weak_areas_percentage", 0) > 18:
                current = self._apply_stage(current, "inpainting", weak_mask, initial_diag)

        # 3. تطبيق كل المراحل بالترتيب
        for stage_name in levels:
            current = self._apply_stage(
                img=current,
                stage=stage_name,
                weak_mask=weak_mask,
                diagnostics=initial_diag
            )

        # 4. التشخيص النهائي
        final_diag = self._extract_diagnostics(current)
        self.final_stats = final_diag.copy()

        # 5. ملخص تحسن في الـ log
        self._log_improvement_summary(initial_diag, final_diag)

        if return_stats:
            return current, final_diag

        return current
    

    def _log_improvement_summary(self, initial: Dict, final: Dict):
        """تسجيل ملخص التحسن بين البداية والنهاية"""
        if not initial or not final:
            return

        i, f = initial, final
        var_change = (f["variance"] / i["variance"] - 1) * 100 if i["variance"] > 0 else 0
        sharp_change = (f["sharpness"] / i["sharpness"] - 1) * 100 if i["sharpness"] > 0 else 0
        clip_change = f["clip_score"] - i["clip_score"]

        logger.info(
            f"ملخص التحسن:\n"
            f"  variance: {i['variance']:.0f} → {f['variance']:.0f}  ({var_change:+.1f}%)\n"
            f"  sharpness: {i['sharpness']:.1f} → {f['sharpness']:.1f}  ({sharp_change:+.1f}%)\n"
            f"  clip_score: {i['clip_score']:.3f} → {f['clip_score']:.3f}  ({clip_change:+.3f})\n"
            f"  weak areas: {i['weak_areas_percentage']:.1f}% → {f['weak_areas_percentage']:.1f}%"
        )
        

    def dna_inspired_single_revival(
            self,
            img: Image.Image,
            mutation_intensity: float = None,  # لو null → استخدم self.mutation_rate
            weak_mask: Optional[np.ndarray] = None,
            use_lpips_guidance: bool = True,
            ref_img: Optional[Image.Image] = None,
        ) -> Image.Image:
            """
            إعادة إحياء جيني سريع ومركز (طفرة واحدة فقط)
            """
            if mutation_intensity is None:
                mutation_intensity = self.mutation_rate

            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32)

            # إنشاء ماسك إذا لم يُمرر
            if weak_mask is None or weak_mask.shape != (img.height, img.width):
                gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                weak_mask = (np.abs(lap) < self.var_threshold).astype(np.uint8) * 255

            # الطفرة الجينية الواحدة المركزة
            h, w, _ = arr.shape
            prob = (weak_mask.astype(float) / 255.0) ** 1.6   # تركيز أقوى في الضعيفة
            prob = np.clip(prob * 3.2 + 0.09, 0.09, 1.0)

            change_mask = np.random.rand(h, w) < prob

            if change_mask.any():
                noise_scale = np.where(change_mask, mutation_intensity * 48, mutation_intensity * 9)
                noise = np.random.normal(0, noise_scale[..., None], size=(h, w, 3))

                # إحياء جيني: إضافة طفرة + دفع نحو الجيران (revival boost)
                neighbors = cv2.GaussianBlur(arr, (7, 7), 0)
                revival_boost = (neighbors - arr) * 0.55 * (weak_mask[..., None] / 255.0)
                arr[change_mask] += noise[change_mask] + revival_boost[change_mask]

                arr = np.clip(arr, 0, 255).astype(np.uint8)

            revived = Image.fromarray(arr)

            # تلميع نهائي خفيف (جزء من الإحياء)
            revived = revived.filter(ImageFilter.SHARPEN)
            revived = ImageEnhance.Contrast(revived).enhance(1.10)
            revived = ImageEnhance.Color(revived).enhance(1.06)

            # توجيه LPIPS (اختياري – بدون تكرار)
            if use_lpips_guidance and self.lpips_loss and ref_img:
                ref_tensor = torch.tensor(np.array(ref_img)).permute(2,0,1).unsqueeze(0).float() / 255.0
                rev_tensor = torch.tensor(np.array(revived)).permute(2,0,1).unsqueeze(0).float() / 255.0
                if self.device.startswith("cuda"):
                    ref_tensor = ref_tensor.cuda()
                    rev_tensor = rev_tensor.cuda()
                dist = self.lpips_loss(ref_tensor, rev_tensor).item()
                if dist > 0.28:  # إذا النتيجة بعيدة → رجوع جزئي للأصل
                    revived = Image.blend(img, revived, 0.65)

            return revived
        
    
    # ────────────────────────────────────────────────
    #          التقارير + الحفظ + المقارنات البصرية
    # ────────────────────────────────────────────────
    def generate_report(self) -> str:
        """
        إنشاء نص تقرير مقروء يلخص التحسن قبل وبعد
        """
        if not self.initial_stats or not self.final_stats:
            return "لا توجد إحصائيات كافية لإنشاء تقرير (لم يتم تشغيل rehabilitate بعد)"

        i = self.initial_stats
        f = self.final_stats

        lines = []
        lines.append("┌──────────────────────────────────────┐")
        lines.append("│      تقرير إعادة التأهيل           │")
        lines.append("└──────────────────────────────────────┘")
        lines.append("")
        lines.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"الـ Prompt: {self.prompt or 'غير محدد'}")
        lines.append("")

        lines.append("قبل التحسين:")
        lines.append(f"  • Variance     : {i.get('variance', 0):.0f}")
        lines.append(f"  • Sharpness    : {i.get('sharpness', 0):.1f}")
        lines.append(f"  • CLIP score   : {i.get('clip_score', 0):.3f}")
        lines.append(f"  • Weak areas   : {i.get('weak_areas_percentage', 0):.1f}%")
        lines.append(f"  • Faces detected: {i.get('faces_detected', 0)}")
        lines.append("")

        lines.append("بعد التحسين:")
        lines.append(f"  • Variance     : {f.get('variance', 0):.0f}   ({(f.get('variance',0)/i.get('variance',1)-1)*100 if i.get('variance',0)>0 else 0:+.1f}%)")
        lines.append(f"  • Sharpness    : {f.get('sharpness', 0):.1f}   ({(f.get('sharpness',0)/i.get('sharpness',1)-1)*100 if i.get('sharpness',0)>0 else 0:+.1f}%)")
        lines.append(f"  • CLIP score   : {f.get('clip_score', 0):.3f}   ({f.get('clip_score',0)-i.get('clip_score',0):+.3f})")
        lines.append(f"  • Weak areas   : {f.get('weak_areas_percentage', 0):.1f}%   ({i.get('weak_areas_percentage',0)-f.get('weak_areas_percentage',0):+.1f}%)")
        lines.append(f"  • Faces detected: {f.get('faces_detected', 0)}")
        lines.append("")

        lines.append("ملاحظات عامة:")
        lines.append("• تم التركيز التلقائي على المناطق الضعيفة")
        if self.use_clip and f.get('clip_score', 0) < 0.30:
            lines.append("• CLIP score منخفض (< 0.30) → قد يكون من الأفضل إعادة التوليد الأساسي")
        lines.append("")

        return "\n".join(lines)


    def save_report(self, filename: Optional[str] = None) -> str:
        """
        حفظ التقرير كنص عادي (.txt)
        """
        report_text = self.generate_report()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.report_dir, f"rehab_report_{timestamp}.txt")

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"تم حفظ التقرير: {filename}")
        except Exception as e:
            logger.error(f"فشل حفظ التقرير: {str(e)}")
            return ""

        return filename


    def save_report_as_pdf(self, filename: Optional[str] = None) -> str:
        """
        محاولة حفظ التقرير كـ PDF (يتطلب reportlab)
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError:
            logger.warning("reportlab غير مثبت → لا يمكن حفظ PDF. قم بـ pip install reportlab")
            return ""

        report_text = self.generate_report()
        lines = report_text.split('\n')

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.report_dir, f"rehab_report_{timestamp}.pdf")

        try:
            c = canvas.Canvas(filename, pagesize=letter)
            width, height = letter
            y = height - 60

            c.setFont("Helvetica", 12)
            for line in lines:
                if y < 50:
                    c.showPage()
                    y = height - 60
                    c.setFont("Helvetica", 12)
                c.drawString(50, y, line)
                y -= 14

            c.save()
            logger.info(f"تم حفظ التقرير كـ PDF: {filename}")
        except Exception as e:
            logger.error(f"فشل إنشاء PDF: {str(e)}")
            return ""

        return filename


    def create_visual_diff(
        self,
        before: Image.Image,
        after: Image.Image,
        filename: Optional[str] = None
    ) -> str:
        """
        إنشاء صورة مقارنة جانب بجانب (قبل | بعد)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.report_dir, f"visual_diff_{timestamp}.png")

        # التأكد من نفس الحجم
        target_size = before.size
        after_resized = after.resize(target_size, Image.Resampling.LANCZOS)

        # إنشاء صورة جديدة مزدوجة العرض
        diff_img = Image.new('RGB', (target_size[0] * 2, target_size[1]))
        diff_img.paste(before, (0, 0))
        diff_img.paste(after_resized, (target_size[0], 0))

        # إضافة نصوص توضيحية
        draw = ImageDraw.Draw(diff_img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        draw.text((20, 20), "قبل التحسين", fill="white", font=font, stroke_width=2, stroke_fill="black")
        draw.text((target_size[0] + 20, 20), "بعد التحسين", fill="white", font=font, stroke_width=2, stroke_fill="black")

        try:
            diff_img.save(filename, quality=95)
            logger.info(f"تم حفظ المقارنة البصرية: {filename}")
        except Exception as e:
            logger.error(f"فشل حفظ visual diff: {str(e)}")
            return ""

        return filename


    def receive_design_report(self, report_text: str, auto_save: bool = True) -> str:
        """
        استقبال تقرير خارجي (مثلاً من design_reporter) وتخزينه
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "timestamp": timestamp,
            "report": report_text
        }
        self.report_history.append(entry)

        # ملخص سريع
        summary_lines = report_text.split('\n')[:8]
        summary = "ملخص التقرير الجديد:\n" + "\n".join(f"  {line.strip()}" for line in summary_lines if line.strip())
        if len(report_text.split('\n')) > 8:
            summary += "\n  ... (التقرير الكامل محفوظ)"

        logger.info(summary)

        if auto_save:
            filename = os.path.join(
                self.report_dir,
                f"design_report_{timestamp.replace(' ', '_').replace(':', '-')}.txt"
            )
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"تقرير خارجي - {timestamp}\n\n")
                    f.write(report_text)
                logger.info(f"تم حفظ تقرير التصميم تلقائيًا: {filename}")
            except Exception as e:
                logger.error(f"فشل حفظ design report: {e}")

        return summary
    
    
    # ────────────────────────────────────────────────
    #     دالة مساعدة متبقية (إذا لم تكن موجودة سابقًا)
    # ────────────────────────────────────────────────
    def _log_improvement_summary(self, initial: Dict, final: Dict):
        """تسجيل ملخص تحسن مختصر في الـ logger (تُستدعى من rehabilitate)"""
        if not initial or not final:
            return

        i, f = initial, final

        var_pct = (f.get('variance', 0) / i.get('variance', 1) - 1) * 100 if i.get('variance', 0) > 0 else 0
        sharp_pct = (f.get('sharpness', 0) / i.get('sharpness', 1) - 1) * 100 if i.get('sharpness', 0) > 0 else 0
        clip_delta = f.get('clip_score', 0) - i.get('clip_score', 0)
        weak_delta = i.get('weak_areas_percentage', 0) - f.get('weak_areas_percentage', 0)

        logger.info(
            "┌────────────── ملخص التحسن ──────────────┐\n"
            f"  Variance    : {var_pct:+.1f}%     ({i.get('variance',0):.0f} → {f.get('variance',0):.0f})\n"
            f"  Sharpness   : {sharp_pct:+.1f}%     ({i.get('sharpness',0):.1f} → {f.get('sharpness',0):.1f})\n"
            f"  CLIP score  : {clip_delta:+.3f}     ({i.get('clip_score',0):.3f} → {f.get('clip_score',0):.3f})\n"
            f"  Weak areas  : {weak_delta:+.1f}%    ({i.get('weak_areas_percentage',0):.1f}% → {f.get('weak_areas_percentage',0):.1f}%)\n"
            "└──────────────────────────────────────────┘"
        )


# ────────────────────────────────────────────────────────────────
#                        مثال التشغيل
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from PIL import Image

    # مثال بسيط للاختبار
    try:
        input_image = Image.open("input_test.png")   # ← غيّر المسار لصورة موجودة عندك
    except FileNotFoundError:
        print("لم يتم العثور على input_test.png")
        print("ضع صورة باسم input_test.png في نفس المجلد أو غيّر المسار")
        sys.exit(1)

    engine = RehabilitationEngineFinal(
        prompt="صورة واقعية عالية الجودة لنسر ذهبي يطير فوق جبال مغطاة بالثلج",
        use_clip=True,
        iteration_count=4,
        mutation_rate=0.012,
        report_dir="my_rehab_reports"
    )

    print("جاري تشغيل إعادة التأهيل في الوضع full...")
    final_image = engine.rehabilitate(
        image=input_image,
        mode="full",
        return_diagnostics=False
    )

    final_image.save("output_rehabbed.png")
    print("تم حفظ الصورة النهائية: output_rehabbed.png")

    # حفظ تقرير تلقائي
    engine.save_report()

    # إنشاء مقارنة بصرية
    engine.create_visual_diff(input_image, final_image)

    print("تم الانتهاء من المثال.")   