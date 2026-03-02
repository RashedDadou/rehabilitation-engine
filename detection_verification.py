# detection_verification.py
"""
كلاس DetectionAndVerification
منفذ كشف وتحديد وتقرير + منفذ ربط مع محرك RehabilitationEngine
"""

import logging
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DetectionAndVerificationInterface:
    """
    المنفذ الداخلي: كشف العيوب، تحديد التحسينات، كتابة التقرير
    """
class DetectionVerificationManager:
    """
    الكلاس الرئيسي لإدارة الكشف والتدقيق وربطه مع محرك التوليد/التأهيل
    """

    def __init__(self, report_dir: str = "reports"):
        self.report_dir = report_dir
        self.history = []  # تاريخ الكشف والتقارير

    def analyze_image_for_issues(self, img: Image.Image) -> Dict[str, float]:
        """
        كشف العيوب في الصورة (blur, noise, asymmetry, color imbalance, text clarity)
        """
        array = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        
        issues = {}
        
        # Blur score
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        issues["blur"] = 1 - (np.var(lap) / 1000)
        issues["blur"] = min(max(issues["blur"], 0.0), 1.0)
        
        # Noise score
        issues["noise"] = np.std(gray) / 255
        issues["noise"] = min(max(issues["noise"], 0.0), 1.0)
        
        # Asymmetry score
        h, w = gray.shape
        left = gray[:, :w//2]
        right = gray[:, w//2:][:, ::-1]
        issues["asymmetry"] = np.mean(np.abs(left - right)) / 255
        
        # Color imbalance
        color_var = [np.var(array[:,:,i]) for i in range(3)]
        issues["color_imbalance"] = np.std(color_var) / np.mean(color_var) if np.mean(color_var) > 0 else 0.0
        
        # Text clarity (edge variance)
        edges = cv2.Canny(gray, 50, 150)
        issues["text_clarity"] = 1 - (np.var(edges) / 10000)
        
        logger.info("كشف العيوب: " + ", ".join([f"{k}: {v:.2f}" for k, v in issues.items()]))
        return issues

    def recommend_improvements(self, issues: Dict[str, float]) -> List[str]:
        """
        تحديد التحسينات المطلوبة بناءً على الكشف
        """
        levels = ["basic"]

        if issues["blur"] > 0.4 or issues["noise"] > 0.4:
            levels.append("advanced")

        if issues["asymmetry"] > 0.3:
            levels.append("symmetry_enhance")

        if issues["color_imbalance"] > 0.3:
            levels.append("color_balance")

        if issues["text_clarity"] > 0.4:
            levels.append("text_enhance")

        if issues["blur"] > 0.5 or issues["noise"] > 0.5:
            levels.append("inpainting")

        levels.append("genetic")  # دائمًا في النهاية

        logger.info(f"التحسينات المقترحة: {levels}")
        return levels

    def generate_report(self, issues_before: Dict[str, float], issues_after: Dict[str, float]) -> str:
        """
        كتابة تقرير دقيق عن عمليات الإصلاح
        """
        report = f"تقرير الكشف والإصلاح - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 50 + "\n\n"

        report += "العيوب قبل الإصلاح:\n"
        for k, v in issues_before.items():
            report += f"• {k.capitalize()}: {v:.2f}\n"

        report += "\nالعيوب بعد الإصلاح:\n"
        for k, v in issues_after.items():
            improvement = ((issues_before[k] - v) / issues_before[k] * 100) if issues_before[k] > 0 else 0
            report += f"• {k.capitalize()}: {v:.2f} (تحسن: {improvement:.1f}%)\n"

        avg_improvement = np.mean([(issues_before[k] - issues_after[k]) / issues_before[k] * 100 
                                  for k in issues_before if issues_before[k] > 0])
        report += f"\nإجمالي التحسن المتوسط: {avg_improvement:.1f}%\n"

        logger.info("تم توليد التقرير")
        return report

    def connect_to_engine(self, engine: Any, image: Image.Image) -> Dict:
        """
        منفذ الربط مع محرك التأهيل (RehabilitationEngineFinal أو أي محرك مشابه)
        - يكشف → يحدد → ينفذ → يرجع الصورة + التقرير
        """
        # 1. كشف العيوب
        issues_before = self.analyze_image_for_issues(image)

        # 2. تحديد التحسينات
        recommended_levels = self.recommend_improvements(issues_before)

        # 3. تنفيذ في المحرك
        final_image = engine.rehabilitate(image, levels=recommended_levels)

        # 4. كشف العيوب بعد التحسين
        issues_after = self.analyze_image_for_issues(final_image)

        # 5. توليد التقرير
        report = self.generate_report(issues_before, issues_after)

        return {
            "final_image": final_image,
            "report": report,
            "issues_before": issues_before,
            "issues_after": issues_after,
            "used_levels": recommended_levels,
        }

    def monitor_improvements(self, issues: Dict[str, float]) -> List[str]:
        """تحديد التحسينات المطلوبة"""
        levels = ["basic"]

        if issues["blur"] > 0.4 or issues["noise"] > 0.4:
            levels.append("advanced")

        if issues["asymmetry"] > 0.3:
            levels.append("symmetry_enhance")

        if issues["color_imbalance"] > 0.3:
            levels.append("color_balance")

        if issues["text_clarity"] > 0.4:
            levels.append("text_enhance")

        if issues["blur"] > 0.5 or issues["noise"] > 0.5:
            levels.append("inpainting")

        levels.append("genetic")
        logger.info(f"التحسينات المقترحة: {levels}")
        return levels

    def design_reporter(self, issues_before: Dict[str, float], issues_after: Dict[str, float]) -> str:
        """تقرير دقيق عن الإصلاح"""
        report = "┌───────────────────────────────┐\n"
        report += "│      تقرير عمليات الإصلاح     │\n"
        report += "└───────────────────────────────┘\n\n"

        report += "العيوب قبل:\n"
        for k, v in issues_before.items():
            report += f"• {k}: {v:.2f}\n"

        report += "\nالعيوب بعد:\n"
        for k, v in issues_after.items():
            improvement = ((issues_before[k] - v) / issues_before[k] * 100) if issues_before[k] > 0 else 0
            report += f"• {k}: {v:.2f} (تحسن: {improvement:.1f}%)\n"

        report += "\nإجمالي التحسن المتوسط: {:.1f}%\n".format(
            np.mean([(issues_before[k] - issues_after[k]) / issues_before[k] * 100 
                     for k in issues_before if issues_before[k] > 0])
        )

        return report


class RehabConnector:
    """
    المنفذ الخارجي: ربط الكشف والتحديد مع محرك RehabilitationEngineFinal
    """
    def __init__(self, engine):
        self.engine = engine
        self.detector = DetectionAndVerificationInterface()

    def process_image(self, image: Image.Image) -> Dict:
        """
        العملية الكاملة: كشف → تحديد → تنفيذ → تقرير
        """
        # 1. كشف العيوب
        issues_before = self.detector.analyze_image_for_issues(image)

        # 2. تحديد التحسينات
        recommended_levels = self.detector.monitor_improvements(issues_before)

        # 3. تنفيذ التحسين في المحرك
        final_image = self.engine.rehabilitate(image, levels=recommended_levels)

        # 4. كشف العيوب بعد التحسين
        issues_after = self.detector.analyze_image_for_issues(final_image)

        # 5. توليد التقرير
        report = self.detector.design_reporter(issues_before, issues_after)

        return {
            "final_image": final_image,
            "report": report,
            "issues_before": issues_before,
            "issues_after": issues_after,
            "used_levels": recommended_levels,
        }