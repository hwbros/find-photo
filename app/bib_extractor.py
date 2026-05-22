import re


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
        results = reader.readtext(image_bytes)
        bibs = set()
        for _, text, confidence in results:
            cleaned = re.sub(r"\s+", "", text)
            if re.fullmatch(r"\d{2,6}", cleaned) and confidence > 0.4:
                bibs.add(cleaned)
        return list(bibs)
