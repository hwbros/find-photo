import io
import re

from PIL import Image, ImageEnhance


def _enhance(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class BibExtractor:
    def __init__(self):
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        return self._reader

    def extract_bibs(self, image_bytes: bytes) -> list[str]:
        reader = self._get_reader()
        enhanced = _enhance(image_bytes)
        results = reader.readtext(enhanced)
        bibs = set()
        for _, text, confidence in results:
            cleaned = re.sub(r"[\s\-_.]", "", text)
            if re.fullmatch(r"\d{2,6}", cleaned) and confidence > 0.3:
                bibs.add(cleaned)
        return list(bibs)
