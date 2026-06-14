"""
Tests for the three new features:

  N1  FileLoaderWorker.progress is emitted while importing TXT and EPUB,
      reaches 100% (curr == total) and reports the right phase label.
  N2  synthesize(combine_progress_callback=...) reports the MP3/merge compilation
      percentage: monotonic non-decreasing, ends at 100, labels carried (MP3+WAV).
  N3  History batch grouping (_group_history_by_batch) buckets a multi-chapter
      render together while leaving old (batch-less) entries standalone.
  N4  _chapter_export_name: sortable, unique, filesystem-safe; strips redundant
      leading numbering from EPUB chapter titles.
  N5  export-as-ZIP writes one entry per chapter, in order, with the right names.

Run with the project venv:
    .venv/Scripts/python.exe tests/test_new_features.py
"""

import os
import sys
import types
import zipfile
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


# --------------------------------------------------------------------------
# Mock pipeline (same shape as kokoro.KPipeline) — no model is loaded.
# --------------------------------------------------------------------------
SAMPLES_PER_CHUNK = 2400  # 0.1s @ 24kHz


class _FakeResult:
    def __init__(self, audio, graphemes):
        self.audio = audio
        self.graphemes = graphemes
        self.phonemes = "phon"


class _FakePipeline:
    def load_voice(self, name):
        return None

    def __call__(self, text, voice=None, speed=1.0):
        yield _FakeResult(torch.full((SAMPLES_PER_CHUNK,), 0.25, dtype=torch.float32), text)


def _make_wrapper(out_dir):
    from tts_wrapper import KokoroTTSWrapper
    w = KokoroTTSWrapper.__new__(KokoroTTSWrapper)
    w.config = {}
    w.output_dir = out_dir
    w.temp_dir = os.path.join(out_dir, "temp_audio")
    w.device = "cpu"
    w.pipeline = _FakePipeline()
    w._encode_progress_cb = None
    os.makedirs(w.output_dir, exist_ok=True)
    os.makedirs(w.temp_dir, exist_ok=True)
    return w


# --------------------------------------------------------------------------
# N1: import progress for TXT and EPUB
# --------------------------------------------------------------------------
def test_import_progress_txt(app):
    print("\nN1a: TXT import emits progress reaching 100%")
    from ui_main import FileLoaderWorker

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "book.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"line {i}" for i in range(250)))

    events = []
    worker = FileLoaderWorker(path)
    worker.progress.connect(lambda c, t, p: events.append((c, t, p)))
    captured = {}
    worker.finished.connect(lambda data: captured.setdefault("data", data))
    worker.error.connect(lambda m: captured.setdefault("error", m))
    worker.run()

    check("error" not in captured, f"no load error (got {captured.get('error')})")
    check(len(events) > 0, f"progress emitted at least once (got {len(events)})")
    check(all(p == "Reading text" for _, _, p in events), "phase label is 'Reading text'")
    last_c, last_t, _ = events[-1]
    check(last_c == last_t and last_t > 0, f"final progress reaches 100% ({last_c}/{last_t})")
    pcts = [int(c / t * 100) for c, t, _ in events]
    check(pcts == sorted(pcts), "percentages are non-decreasing")


def test_import_progress_epub(app):
    print("\nN1b: EPUB import emits progress reaching 100%")
    from ebooklib import epub
    from ui_main import FileLoaderWorker

    tmp = tempfile.mkdtemp()
    epub_path = os.path.join(tmp, "sample.epub")
    book = epub.EpubBook()
    book.set_identifier("id-1")
    book.set_title("Prog Book")
    book.set_language("en")
    chaps = []
    for i in range(1, 5):
        c = epub.EpubHtml(title=f"Ch {i}", file_name=f"c{i}.xhtml", lang="en")
        c.content = f"<html><body><h1>Ch {i}</h1><p>Body {i}.</p></body></html>"
        book.add_item(c)
        chaps.append(c)
    book.toc = tuple(chaps)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chaps
    epub.write_epub(epub_path, book)

    events = []
    worker = FileLoaderWorker(epub_path)
    worker.progress.connect(lambda c, t, p: events.append((c, t, p)))
    captured = {}
    worker.finished.connect(lambda data: captured.setdefault("data", data))
    worker.error.connect(lambda m: captured.setdefault("error", m))
    worker.run()

    check("error" not in captured, f"no load error (got {captured.get('error')})")
    check(len(events) > 0, f"progress emitted ({len(events)} events)")
    check(all(p == "Parsing EPUB" for _, _, p in events), "phase label is 'Parsing EPUB'")
    last_c, last_t, _ = events[-1]
    check(last_c == last_t and last_t > 0, f"final progress reaches total ({last_c}/{last_t})")


# --------------------------------------------------------------------------
# N2: MP3 / merge compilation progress
# --------------------------------------------------------------------------
def _run_combine(out_fmt):
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)
    segments = [(f"line {i}", ["af_heart"], None) for i in range(6)]
    labels = ["A"] * 3 + ["B"] * 3
    events = []
    _results, combined = w.synthesize(
        segments=segments, sample_rate=24000,
        output_format=out_fmt, chapter_labels=labels,
        combine_progress_callback=lambda label, pct: events.append((label, pct)),
    )
    return events, combined


def test_merge_progress_mp3(app):
    print("\nN2a: MP3 compilation reports progress 0..100")
    events, combined = _run_combine("MP3")
    check(len(combined) == 2, f"2 chapter files (got {len(combined)})")
    check(len(events) > 0, f"combine callback fired ({len(events)} events)")
    pcts = [p for _, p in events]
    check(all(0 <= p <= 100 for p in pcts), "all percents within 0..100")
    check(pcts == sorted(pcts), f"percents non-decreasing (got {pcts})")
    check(max(pcts) == 100, f"reaches 100% (got max {max(pcts)})")
    labels = {l for l, _ in events}
    check("A" in labels and "B" in labels, f"both chapter labels reported (got {labels})")


def test_merge_progress_wav(app):
    print("\nN2b: WAV compilation reports per-chapter progress to 100")
    events, combined = _run_combine("WAV")
    check(len(combined) == 2, f"2 chapter files (got {len(combined)})")
    pcts = [p for _, p in events]
    check(pcts == sorted(pcts), f"percents non-decreasing (got {pcts})")
    check(max(pcts) == 100, f"reaches 100% (got {pcts})")


def test_merge_progress_optional(app):
    print("\nN2c: combine_progress_callback is optional (back-compat)")
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)
    segments = [(f"l{i}", ["af_heart"], None) for i in range(3)]
    raised = False
    try:
        _r, combined = w.synthesize(segments=segments, sample_rate=24000, output_format="WAV")
    except Exception:
        raised = True
        combined = []
    check(not raised, "synthesize works without the new callback")
    check(len(combined) == 1, "single combined file produced")


# --------------------------------------------------------------------------
# N3: history batch grouping
# --------------------------------------------------------------------------
def test_batch_grouping(app):
    print("\nN3: _group_history_by_batch buckets a book, leaves others standalone")
    from ui_main import MyTTSMainWindow

    fake = types.SimpleNamespace(synthesis_results=[
        {"combined": "old.wav"},                                  # legacy, no batch_id
        {"combined": "c1.wav", "batch_id": "B1", "chapter_order": 1},
        {"combined": "c2.wav", "batch_id": "B1", "chapter_order": 2},
        {"combined": "c3.wav", "batch_id": "B1", "chapter_order": 3},
        {"combined": "solo.wav", "batch_id": "B2", "chapter_order": 1},  # single-file render
    ])
    groups = MyTTSMainWindow._group_history_by_batch(fake)
    check(len(groups) == 3, f"3 groups: legacy, book(B1), solo(B2) (got {len(groups)})")
    sizes = [len(g) for g in groups]
    check(sizes == [1, 3, 1], f"group sizes [1,3,1] in original order (got {sizes})")
    # The book group keeps real indices 1,2,3
    book = [g for g in groups if len(g) > 1][0]
    check([ri for ri, _ in book] == [1, 2, 3], "book group carries real indices 1,2,3")


# --------------------------------------------------------------------------
# N4: export name builder
# --------------------------------------------------------------------------
def test_chapter_export_name(app):
    print("\nN4: _chapter_export_name is safe, sortable, unique")
    from ui_main import MyTTSMainWindow
    fn = MyTTSMainWindow._chapter_export_name
    used = set()

    n1 = fn(1, "01. The Beginning", "x.mp3", used)
    check(n1 == "01 - The_Beginning.mp3", f"strips leading number, pads order (got {n1})")
    n2 = fn(2, "A New: Dawn?", "y.wav", used)
    check(n2 == "02 - A_New_Dawn.wav", f"sanitizes illegal chars (got {n2})")
    # Duplicate title -> unique suffix
    n3 = fn(3, "01. The Beginning", "z.mp3", used)
    check(n3 != n1 and n3.startswith("03 - "), f"duplicate title disambiguated (got {n3})")
    # Missing title -> Chapter_NN
    n4 = fn(4, None, "w.mp3", used)
    check(n4 == "04 - Chapter_04.mp3", f"missing title falls back (got {n4})")


# --------------------------------------------------------------------------
# N5: ZIP export round-trip
# --------------------------------------------------------------------------
def test_export_zip(app):
    print("\nN5: export-as-ZIP bundles all chapters in order")
    from ui_main import MyTTSMainWindow
    from PySide6.QtWidgets import QFileDialog

    src_dir = tempfile.mkdtemp()
    chapters = []
    for k, title in [(1, "01. Intro"), (2, "02. Body"), (3, "03. Outro")]:
        p = os.path.join(src_dir, f"src_{k}.mp3")
        with open(p, "wb") as f:
            f.write(b"ID3" + bytes([k]) * 100)
        chapters.append((k, title, p))

    zip_path = os.path.join(tempfile.mkdtemp(), "MyBook.zip")
    orig = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (zip_path, "Zip Archive (*.zip)"))
    try:
        fake = types.SimpleNamespace(
            statusBar=lambda: types.SimpleNamespace(showMessage=lambda *a, **k: None),
            _chapter_export_name=MyTTSMainWindow._chapter_export_name,
            _report_export_result=lambda *a, **k: None,
        )
        MyTTSMainWindow._export_book_zip(fake, "MyBook", chapters)
    finally:
        QFileDialog.getSaveFileName = orig

    check(os.path.exists(zip_path), "zip file created")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    check(names == ["01 - Intro.mp3", "02 - Body.mp3", "03 - Outro.mp3"],
          f"all 3 chapters zipped in order with clean names (got {names})")


def test_indices_for_batch(app):
    print("\nN6: _indices_for_batch resolves CURRENT positions (no stale indices)")
    from ui_main import MyTTSMainWindow

    fake = types.SimpleNamespace(synthesis_results=[
        {"batch_id": "B1"}, {"batch_id": "B1"}, {"batch_id": "B2"}, {"batch_id": "B1"},
    ])
    check(MyTTSMainWindow._indices_for_batch(fake, "B1") == [0, 1, 3],
          "finds all B1 entries by id")
    check(MyTTSMainWindow._indices_for_batch(fake, None) == [],
          "None batch_id -> empty (no accidental matches)")
    # Deleting an earlier entry shifts positions; resolving again must be correct.
    del fake.synthesis_results[0]
    check(MyTTSMainWindow._indices_for_batch(fake, "B1") == [0, 2],
          "re-resolves to shifted positions after a deletion")


def test_export_numbering_no_gaps(app):
    print("\nN7: export numbering is sequential even when a chapter is missing")
    from ui_main import MyTTSMainWindow
    from PySide6.QtWidgets import QFileDialog

    src_dir = tempfile.mkdtemp()
    # Simulate chapters where the middle one (order 2) is gone: orders 1 and 3.
    chapters_raw = [(1, "01. A", None), (3, "03. C", None)]
    chapters = []
    for order, title, _ in chapters_raw:
        p = os.path.join(src_dir, f"src_{order}.mp3")
        with open(p, "wb") as f:
            f.write(b"ID3" + bytes([order]) * 50)
        chapters.append((order, title, p))
    # Mimic export_book's resequencing: sort by order, then assign 1..N.
    chapters.sort(key=lambda c: c[0])
    chapters = [(seq, t, pth) for seq, (_o, t, pth) in enumerate(chapters, start=1)]

    zip_path = os.path.join(tempfile.mkdtemp(), "Gapped.zip")
    orig = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (zip_path, ""))
    try:
        fake = types.SimpleNamespace(
            statusBar=lambda: types.SimpleNamespace(showMessage=lambda *a, **k: None),
            _chapter_export_name=MyTTSMainWindow._chapter_export_name,
            _report_export_result=lambda *a, **k: None,
        )
        MyTTSMainWindow._export_book_zip(fake, "Gapped", chapters)
    finally:
        QFileDialog.getSaveFileName = orig

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    check(names == ["01 - A.mp3", "02 - C.mp3"],
          f"sequential 01,02 with no gap despite missing chapter 2 (got {names})")


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    tests = [
        test_import_progress_txt,
        test_import_progress_epub,
        test_merge_progress_mp3,
        test_merge_progress_wav,
        test_merge_progress_optional,
        test_batch_grouping,
        test_chapter_export_name,
        test_export_zip,
        test_indices_for_batch,
        test_export_numbering_no_gaps,
    ]
    for t in tests:
        try:
            t(app)
        except Exception:
            print(f"  [FAIL] {t.__name__} raised:")
            traceback.print_exc()
            _FAILURES.append(f"{t.__name__} raised exception")

    print("\n" + "=" * 60)
    if _FAILURES:
        print(f"RESULT: {len(_FAILURES)} FAILURE(S)")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
