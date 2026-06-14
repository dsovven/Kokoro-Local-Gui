"""
End-to-end tests for the chapter-splitting feature and the audio-save fix.

Run with the project venv:
    .venv/Scripts/python.exe tests/test_chapter_split.py

Covers:
  T1  EPUB loader -> (chapter_title, line) tuples, spine order, nav/cover skipped
  T2  synthesize() -> one combined file per chapter (MP3 + WAV), correct names,
      chunk_indices, and exact audio length (no >4GB WAV-header crash path)
  T3  synthesize() with no chapters -> single combined file, legacy naming
  T4  project persistence round-trip preserves the chapter field
  T5  Qt QTableWidgetItem UserRole round-trip (mechanism used by save/load_project)

No Kokoro model is loaded: the pipeline is mocked. Real ebooklib, torch,
soundfile and ffmpeg are exercised.
"""

import os
import sys
import tempfile
import traceback

# Make repo root importable and force headless Qt.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
import soundfile as sf

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


# --------------------------------------------------------------------------
# Mock pipeline (stands in for kokoro.KPipeline)
# --------------------------------------------------------------------------
SAMPLES_PER_CHUNK = 2400  # 0.1s @ 24kHz


class _FakeResult:
    def __init__(self, audio, graphemes):
        self.audio = audio
        self.graphemes = graphemes
        self.phonemes = "phon"


class _FakePipeline:
    """Yields one deterministic audio chunk per call."""
    def load_voice(self, name):
        return None

    def __call__(self, text, voice=None, speed=1.0):
        audio = torch.full((SAMPLES_PER_CHUNK,), 0.25, dtype=torch.float32)
        yield _FakeResult(audio, text)


def _make_wrapper(out_dir):
    from tts_wrapper import KokoroTTSWrapper
    w = KokoroTTSWrapper.__new__(KokoroTTSWrapper)
    w.config = {}
    w.output_dir = out_dir
    w.temp_dir = os.path.join(out_dir, "temp_audio")
    w.device = "cpu"
    w.pipeline = _FakePipeline()
    os.makedirs(w.output_dir, exist_ok=True)
    os.makedirs(w.temp_dir, exist_ok=True)
    return w


# --------------------------------------------------------------------------
# T1: EPUB parsing
# --------------------------------------------------------------------------
def test_epub_parsing(app):
    print("\nT1: EPUB loader chapter extraction")
    from ebooklib import epub
    from ui_main import FileLoaderWorker

    tmp = tempfile.mkdtemp()
    epub_path = os.path.join(tmp, "sample.epub")

    book = epub.EpubBook()
    book.set_identifier("id-123")
    book.set_title("Sample Book")
    book.set_language("en")

    chapters_data = [
        ("The Beginning", ["Once upon a time.", "It was a dark night."]),
        ("A New Dawn", ["Morning came.", "Birds sang loudly.", "Then silence."]),
        ("The End", ["All was well."]),
    ]
    epub_chapters = []
    for i, (title, paras) in enumerate(chapters_data, 1):
        c = epub.EpubHtml(title=title, file_name=f"chap_{i}.xhtml", lang="en")
        c.content = ("<html><body><h1>" + title + "</h1>"
                     + "".join(f"<p>{p}</p>" for p in paras)
                     + "</body></html>")
        book.add_item(c)
        epub_chapters.append(c)

    book.toc = tuple(epub_chapters)
    book.add_item(epub.EpubNcx())
    nav = epub.EpubNav()
    book.add_item(nav)
    # Realistic spine: nav document first, then chapters. Loader must skip nav.
    book.spine = ["nav"] + epub_chapters
    epub.write_epub(epub_path, book)

    captured = {}
    worker = FileLoaderWorker(epub_path)
    worker.finished.connect(lambda data: captured.setdefault("data", data))
    worker.error.connect(lambda msg: captured.setdefault("error", msg))
    worker.run()

    check("error" not in captured, f"no load error (got: {captured.get('error')})")
    data = captured.get("data", [])
    check(all(isinstance(x, tuple) and len(x) == 2 for x in data),
          "every entry is a (chapter, line) tuple")

    titles_in_order = []
    for chap, _line in data:
        if not titles_in_order or titles_in_order[-1] != chap:
            titles_in_order.append(chap)

    check(titles_in_order == ["01. The Beginning", "02. A New Dawn", "03. The End"],
          f"chapters numbered, in spine order, nav skipped (got {titles_in_order})")

    # Per-chapter line counts: heading line + paragraphs
    from collections import Counter
    counts = Counter(chap for chap, _ in data)
    check(counts.get("01. The Beginning") == 3, "Ch1 has heading + 2 paragraphs = 3 lines")
    check(counts.get("02. A New Dawn") == 4, "Ch2 has heading + 3 paragraphs = 4 lines")
    check(counts.get("03. The End") == 2, "Ch3 has heading + 1 paragraph = 2 lines")

    texts = [line for _, line in data]
    check("Once upon a time." in texts, "paragraph text preserved")
    check(not any("Table of Contents" in t for t in texts),
          "nav/TOC text not present in output")


# --------------------------------------------------------------------------
# T2: synthesize -> per-chapter files
# --------------------------------------------------------------------------
def _read_audio_len(path):
    data, _sr = sf.read(path)
    return len(data)


def test_synth_chapters_mp3(app):
    print("\nT2a: synthesize() per-chapter MP3 output")
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)

    segments = [(f"line {i}", ["af_heart"], None) for i in range(6)]
    labels = ["Chapter One"] * 3 + ["Chapter Two"] * 3

    results, combined = w.synthesize(
        segments=segments, sample_rate=24000,
        output_format="MP3", chapter_labels=labels,
    )

    check(len(combined) == 2, f"2 chapter files produced (got {len(combined)})")
    paths = [c["path"] for c in combined]
    check(all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths),
          "both MP3 files exist and are non-empty")
    names = [os.path.basename(p) for p in paths]
    check(any("Chapter_One" in n for n in names),
          f"file 1 named for chapter (got {names})")
    check(any("Chapter_Two" in n for n in names),
          f"file 2 named for chapter (got {names})")
    check(len(set(names)) == 2, f"filenames are unique (got {names})")
    check(all(n.endswith(".mp3") for n in names), "extension is .mp3")
    check(combined[0]["title"] == "Chapter One", "title carried on result[0]")
    check(combined[0]["chunk_indices"] == [0, 1, 2], "chunk_indices for ch1")
    check(combined[1]["chunk_indices"] == [3, 4, 5], "chunk_indices for ch2")
    check(len(results) == 6, "6 per-chunk results returned")


def test_synth_chapters_wav(app):
    print("\nT2b: synthesize() per-chapter WAV output (exact sample counts)")
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)

    segments = [(f"line {i}", ["af_heart"], None) for i in range(5)]
    labels = ["Intro"] * 2 + ["Body"] * 3

    results, combined = w.synthesize(
        segments=segments, sample_rate=24000,
        output_format="WAV", chapter_labels=labels,
    )

    check(len(combined) == 2, f"2 chapter files produced (got {len(combined)})")
    by_title = {c["title"]: c for c in combined}
    len_intro = _read_audio_len(by_title["Intro"]["path"])
    len_body = _read_audio_len(by_title["Body"]["path"])
    check(len_intro == 2 * SAMPLES_PER_CHUNK,
          f"Intro length = 2 chunks ({len_intro} == {2*SAMPLES_PER_CHUNK})")
    check(len_body == 3 * SAMPLES_PER_CHUNK,
          f"Body length = 3 chunks ({len_body} == {3*SAMPLES_PER_CHUNK})")


def test_synth_consecutive_grouping(app):
    print("\nT2c: non-contiguous identical titles stay as separate groups")
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)
    segments = [(f"l{i}", ["af_heart"], None) for i in range(4)]
    labels = ["A", "B", "A", "A"]  # A appears in two non-adjacent blocks
    _results, combined = w.synthesize(
        segments=segments, sample_rate=24000,
        output_format="WAV", chapter_labels=labels,
    )
    check(len(combined) == 3, f"3 groups (A,B,A) -> 3 files (got {len(combined)})")
    check([c["chunk_indices"] for c in combined] == [[0], [1], [2, 3]],
          "grouping follows consecutive runs")
    # Filenames must be unique even with repeated titles
    names = [os.path.basename(c["path"]) for c in combined]
    check(len(set(names)) == 3, f"filenames unique despite repeated title (got {names})")


# --------------------------------------------------------------------------
# T3: no chapters -> single legacy file
# --------------------------------------------------------------------------
def test_synth_no_chapters(app):
    print("\nT3: synthesize() without chapters -> single file, legacy name")
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)
    segments = [(f"line {i}", ["af_heart"], None) for i in range(4)]

    results, combined = w.synthesize(
        segments=segments, sample_rate=24000, output_format="WAV",
        chapter_labels=None,
    )
    check(len(combined) == 1, f"single combined file (got {len(combined)})")
    name = os.path.basename(combined[0]["path"])
    check(name.startswith("combined_") and "_ch" not in name,
          f"legacy filename, no chapter suffix (got {name})")
    check(combined[0]["title"] is None, "title is None")
    check(combined[0]["chunk_indices"] == [0, 1, 2, 3], "all chunks in one group")
    total = _read_audio_len(combined[0]["path"])
    check(total == 4 * SAMPLES_PER_CHUNK, "full audio length preserved")


# --------------------------------------------------------------------------
# T4: project persistence round-trip with chapter
# --------------------------------------------------------------------------
def test_project_persistence(app):
    print("\nT4: project save/load preserves chapter field")
    import persistence
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "proj.kproj")

    data = [
        {"text": "Line A", "v1": "X", "v2": "None (Single Voice)", "chapter": "Ch 1"},
        {"text": "Line B", "v1": "X", "v2": "None (Single Voice)", "chapter": "Ch 1"},
        {"text": "Line C", "v1": "Y", "v2": "None (Single Voice)"},  # no chapter
    ]
    ok = persistence.save_project_file(path, data)
    check(ok, "save_project_file returned True")
    loaded = persistence.load_project_file(path)
    check(loaded == data, "round-trip is loss-less (chapter survives)")
    check(loaded[0].get("chapter") == "Ch 1", "chapter field present after reload")
    check(loaded[2].get("chapter") is None, "chapter absent stays absent")


# --------------------------------------------------------------------------
# T5: Qt UserRole round-trip (the mechanism save_project/load_project rely on)
# --------------------------------------------------------------------------
def test_qt_userrole(app):
    print("\nT5: QTableWidgetItem UserRole stores/returns chapter")
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
    from PySide6.QtCore import Qt

    table = QTableWidget(2, 1)
    it0 = QTableWidgetItem("hello")
    it0.setData(Qt.ItemDataRole.UserRole, "My Chapter")
    table.setItem(0, 0, it0)
    it1 = QTableWidgetItem("world")
    it1.setData(Qt.ItemDataRole.UserRole, None)
    table.setItem(1, 0, it1)

    check(table.item(0, 0).data(Qt.ItemDataRole.UserRole) == "My Chapter",
          "UserRole returns stored chapter")
    check(table.item(1, 0).data(Qt.ItemDataRole.UserRole) is None,
          "UserRole None stays None")


def test_same_heading_not_merged(app):
    """Regression (Bug B): two consecutive EPUB docs sharing a heading must
    still become two separate chapter files, not one merged file."""
    print("\nT6: distinct EPUB docs with identical heading stay separate")
    from ebooklib import epub
    from ui_main import FileLoaderWorker

    tmp = tempfile.mkdtemp()
    epub_path = os.path.join(tmp, "dup.epub")
    book = epub.EpubBook()
    book.set_identifier("dup-1")
    book.set_title("Dup Headings")
    book.set_language("en")

    chaps = []
    bodies = [["Alpha content one."], ["Beta content two."], ["Gamma content three."]]
    for i, paras in enumerate(bodies, 1):
        c = epub.EpubHtml(title="Chapter", file_name=f"c{i}.xhtml", lang="en")
        # Every document uses the SAME heading text on purpose.
        c.content = "<html><body><h1>Chapter</h1>" + "".join(f"<p>{p}</p>" for p in paras) + "</body></html>"
        book.add_item(c)
        chaps.append(c)
    book.toc = tuple(chaps)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chaps
    epub.write_epub(epub_path, book)

    captured = {}
    worker = FileLoaderWorker(epub_path)
    worker.finished.connect(lambda data: captured.setdefault("data", data))
    worker.run()
    data = captured.get("data", [])

    labels = []
    for chap, _ in data:
        if not labels or labels[-1] != chap:
            labels.append(chap)
    check(len(labels) == 3, f"3 distinct labels despite identical headings (got {labels})")
    check(len(set(labels)) == 3, "labels are unique")

    # Feed through synthesize and confirm 3 separate files result.
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)
    segments = [(line, ["af_heart"], None) for _, line in data]
    chapter_labels = [chap for chap, _ in data]
    _results, combined = w.synthesize(
        segments=segments, sample_rate=24000,
        output_format="WAV", chapter_labels=chapter_labels,
    )
    check(len(combined) == 3, f"3 chapter files produced, not merged (got {len(combined)})")
    paths = [c["path"] for c in combined]
    check(len(set(paths)) == 3, "3 unique output paths")


def test_partial_save_failure(app):
    """Regression (Bug A): if one chapter fails to save, the others must still
    be returned (not discarded), and an all-fail run raises."""
    print("\nT7: partial chapter save failure keeps successful chapters")
    out = tempfile.mkdtemp()
    w = _make_wrapper(out)
    segments = [(f"line {i}", ["af_heart"], None) for i in range(6)]
    labels = ["A"] * 2 + ["B"] * 2 + ["C"] * 2  # 3 chapters

    orig = w.save_audio
    state = {"combined": 0}

    def flaky(audio, filepath, format="WAV", target_sample_rate=24000):
        # Only interfere with the per-chapter combined files; let chunk saves pass.
        if os.path.basename(filepath).startswith("combined_"):
            state["combined"] += 1
            if state["combined"] == 2:  # fail the 2nd chapter ("B")
                raise RuntimeError("simulated save failure")
        return orig(audio, filepath, format=format, target_sample_rate=target_sample_rate)

    w.save_audio = flaky
    raised = False
    try:
        _results, combined = w.synthesize(
            segments=segments, sample_rate=24000,
            output_format="WAV", chapter_labels=labels,
        )
    except Exception:
        raised = True
        combined = []
    check(not raised, "run does not abort on a single chapter failure")
    titles = sorted(c["title"] for c in combined)
    check(titles == ["A", "C"], f"surviving chapters A and C returned, B dropped (got {titles})")
    check(all(os.path.exists(c["path"]) for c in combined), "survivor files exist on disk")

    # All-fail case -> RuntimeError
    out2 = tempfile.mkdtemp()
    w2 = _make_wrapper(out2)
    orig2 = w2.save_audio

    def always_fail(audio, filepath, format="WAV", target_sample_rate=24000):
        if os.path.basename(filepath).startswith("combined_"):
            raise RuntimeError("boom")
        return orig2(audio, filepath, format=format, target_sample_rate=target_sample_rate)

    w2.save_audio = always_fail
    all_failed_raised = False
    try:
        w2.synthesize(segments=segments, sample_rate=24000,
                      output_format="WAV", chapter_labels=labels)
    except RuntimeError:
        all_failed_raised = True
    check(all_failed_raised, "run raises when every chapter fails to save")


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    tests = [
        test_epub_parsing,
        test_synth_chapters_mp3,
        test_synth_chapters_wav,
        test_synth_consecutive_grouping,
        test_synth_no_chapters,
        test_project_persistence,
        test_qt_userrole,
        test_same_heading_not_merged,
        test_partial_save_failure,
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
