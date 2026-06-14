import os
import re
import time
import threading
import subprocess
from collections import deque
import torch
import numpy as np
import soundfile as sf
import logging
import librosa # Essential for pitch shifting!

from typing import Optional, List, Tuple, Callable
from pydub import AudioSegment
from kokoro import KPipeline 
from models import build_pipeline, list_available_voices, get_internal_voice_name

# Use the logger configured in main.py
logger = logging.getLogger(__name__)

# Define constants
OUTPUTS_DIR = "outputs"
TEMP_DIR = "temp_audio"
CHUNK_PREFIX = "chunk_"
DEFAULT_SAMPLERATE = 24000

def _sanitize_filename(name: str, max_len: int = 50) -> str:
    """Make a chapter title safe to use inside a filename."""
    name = re.sub(r'[<>:"/\\|?*\n\r\t]+', ' ', name or "")
    name = re.sub(r'\s+', '_', name).strip('._')
    return name[:max_len] or "chapter"

class KokoroTTSWrapper:
    """Wraps Kokoro KPipeline, handles voice loading, blending with weights, saving."""

    def __init__(
        self,
        output_dir: str = OUTPUTS_DIR,
        temp_sub_dir: str = TEMP_DIR,
        config: Optional[dict] = None
    ):
        logger.info("KokoroTTSWrapper.__init__ START")
        self.config = config if config else {}
        self.output_dir = output_dir
        self.temp_dir = os.path.join(self.output_dir, temp_sub_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        logger.info(f"TTS Wrapper using device: {self.device}")
        self.pipeline: Optional[KPipeline] = None
        # Per-chapter ffmpeg progress hook, set transiently by synthesize().
        self._encode_progress_cb: Optional[Callable[[float], None]] = None

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        try:
            # Build the pipeline on initialization
            self.pipeline = build_pipeline(device=self.device)
            logger.info(f"Kokoro Pipeline built successfully on device {self.device}.")
        except Exception as e:
            logger.exception("Failed to initialize Kokoro pipeline.")
            raise RuntimeError(f"Failed to initialize TTS engine: {e}") from e

        logger.info("KokoroTTSWrapper.__init__ END")

    def synthesize(
        self,
        segments: List[Tuple[str, List[str], Optional[str]]],
        speed: float = 1.0,
        pitch: float = 1.0,
        alpha: float = 0.0, 
        beta: float = 0.0,  
        diffusion_steps: int = 0,
        embedding_scale: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLERATE,
        output_format: str = 'WAV',
        chapter_labels: Optional[List[Optional[str]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        combine_progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Tuple[List[Tuple[str, str, np.ndarray, str]], List[dict]]:
        
        if not self.pipeline:
            raise RuntimeError("TTS Pipeline is not initialized.")

        logger.info(f"Starting synthesis. Speed: {speed}, Pitch: {pitch}, Rate: {sample_rate}")

        all_audio_tensors: List[torch.Tensor] = []
        synthesis_result_list: List[Tuple[str, str, np.ndarray, str]] = []
        chunk_chapter_labels: List[Optional[str]] = []
        combined_filepaths: List[dict] = []
        total_segments = len(segments)

        try:
            # --- 1. Pre-load Voices (Standardized) ---
            # We must resolve all friendly names to internal names first
            voice_map = {}
            unique_internal_voices_needed = set()

            for _, segment_voices, _ in segments:
                for v_str in segment_voices:
                    # Handle "VoiceA+VoiceB" from UI
                    parts = v_str.split('+')
                    resolved_parts = []
                    for part in parts:
                        clean_part = part.split(':')[0].strip() # Remove weights if any
                        internal = get_internal_voice_name(clean_part)
                        if not internal:
                            # Fallback for custom files
                            internal = clean_part.lower().replace(" ", "_")
                        
                        unique_internal_voices_needed.add(internal)
                        resolved_parts.append(internal)
                    
                    # Map the full UI string to a COMMA-separated internal string
                    # e.g., "Alice+Bob" -> "bf_alice,bm_bob"
                    voice_map[v_str] = ",".join(resolved_parts)

            logger.info(f"Pre-loading {len(unique_internal_voices_needed)} voices: {unique_internal_voices_needed}")
            for internal_name in unique_internal_voices_needed:
                try:
                    self.pipeline.load_voice(internal_name)
                except Exception as load_error:
                    logger.error(f"Failed to load voice {internal_name}: {str(load_error)}")
                    pass 

            # --- 2. Process Segments ---
            for i, (text_chunk, segment_voices, weight_str) in enumerate(segments):
                segment_num = i + 1
                current_chapter = (
                    chapter_labels[i] if chapter_labels and i < len(chapter_labels) else None
                )
                if not text_chunk.strip():
                    continue

                # Get the comma-separated spec we built earlier
                # If segment_voices is ['Alice+Bob'], we get 'bf_alice,bm_bob'
                # If multiple entries (rare in your UI logic), join them with comma too
                internal_specs = [voice_map.get(v, v) for v in segment_voices]
                blended_voice_spec = ",".join(internal_specs)
                
                logger.debug(f"Segment {segment_num}: Voice spec passed to pipeline: '{blended_voice_spec}'")
                
                chunk_results_count = 0
                try:
                    # Kwargs for pipeline
                    generate_kwargs = {
                        "voice": blended_voice_spec,
                        "speed": speed,
                    }
                    
                    # Advanced params logging
                    if diffusion_steps > 0 or alpha > 0 or beta > 0:
                        logger.debug("Advanced style params ignored (not supported by current Kokoro build).")

                    for result in self.pipeline(text_chunk, **generate_kwargs):
                        if hasattr(result, 'audio') and result.audio is not None:
                            try:
                                audio_tensor = result.audio.cpu().float().squeeze()
                                if audio_tensor.ndim != 1: 
                                    audio_tensor = audio_tensor.flatten()
                                
                                audio_data_numpy = audio_tensor.numpy()

                                # --- Pitch Shift ---
                                if pitch != 1.0:
                                    try:
                                        n_steps = 12 * np.log2(pitch)
                                        audio_data_numpy = librosa.effects.pitch_shift(
                                            audio_data_numpy, 
                                            sr=DEFAULT_SAMPLERATE, 
                                            n_steps=n_steps
                                        )
                                        audio_tensor = torch.from_numpy(audio_data_numpy)
                                    except Exception as e_pitch:
                                        logger.error(f"Pitch shift failed: {e_pitch}")

                                chunk_timestamp = time.strftime("%Y%m%d_%H%M%S")
                                unique_suffix = f"{segment_num}_{chunk_results_count}_{int(time.time()*1000)}"
                                chunk_filepath = os.path.join(self.temp_dir, f"{CHUNK_PREFIX}{chunk_timestamp}_{unique_suffix}.wav")
                                
                                # Save Chunk (WAV, Resampled)
                                self.save_audio(audio_data_numpy, chunk_filepath, format='WAV', target_sample_rate=sample_rate)

                                graphemes = getattr(result, 'graphemes', None) or ""
                                phonemes = getattr(result, 'phonemes', None) or ""
                                synthesis_result_list.append((graphemes, phonemes, audio_data_numpy, chunk_filepath))
                                all_audio_tensors.append(audio_tensor)
                                chunk_chapter_labels.append(current_chapter)
                                chunk_results_count += 1
                                
                            except Exception as proc_err:
                                logger.exception(f"Error processing chunk for seg {segment_num}: {proc_err}")
                                continue

                except Exception as synth_call_err:
                    logger.exception(f"Pipeline error seg {segment_num}: {synth_call_err}")
                    raise

                if progress_callback:
                    progress_callback(segment_num, total_segments)

            # --- 3. Final Combination (grouped by chapter) ---
            if all_audio_tensors:
                # Group consecutive chunks that share the same chapter label.
                # Chapters come in reading order, so each chapter's chunks are contiguous.
                groups: List[dict] = []
                for idx, (tensor, label) in enumerate(zip(all_audio_tensors, chunk_chapter_labels)):
                    if groups and groups[-1]["label"] == label:
                        groups[-1]["tensors"].append(tensor)
                        groups[-1]["chunk_indices"].append(idx)
                    else:
                        groups.append({"label": label, "tensors": [tensor], "chunk_indices": [idx]})

                multi = len(groups) > 1
                combined_timestamp = time.strftime("%Y%m%d_%H%M%S")
                ext = output_format.lower()
                logger.info(
                    f"Combining {len(all_audio_tensors)} audio chunks into "
                    f"{len(groups)} file(s)..."
                )

                total_groups = len(groups)
                save_failures = 0
                used_paths = set()
                for chapter_num, group in enumerate(groups, start=1):
                    # Save each chapter independently. A single chapter's save
                    # failure must NOT discard the chapters already written, so we
                    # catch per chapter and keep going.
                    chapter_label = group["label"] or f"Chapter {chapter_num:02d}"

                    # Report progress at the START of this chapter's encode, and
                    # install a per-chapter ffmpeg callback that maps the encoder's
                    # 0..1 fraction onto the overall 0..100 compilation progress.
                    if combine_progress_callback:
                        combine_progress_callback(
                            chapter_label, int((chapter_num - 1) / total_groups * 100)
                        )
                        self._encode_progress_cb = (
                            lambda frac, _k=chapter_num, _n=total_groups, _lbl=chapter_label:
                            combine_progress_callback(
                                _lbl, int(min(((_k - 1) + frac) / _n, 1.0) * 100)
                            )
                        )
                    else:
                        self._encode_progress_cb = None

                    try:
                        combined_audio_numpy = torch.cat(group["tensors"], dim=0).cpu().float().numpy()

                        if multi:
                            safe = _sanitize_filename(group["label"]) if group["label"] else f"chapter_{chapter_num:02d}"
                            combined_filename = f"combined_{combined_timestamp}_{safe}.{ext}"
                        else:
                            combined_filename = f"combined_{combined_timestamp}.{ext}"
                        combined_filepath = os.path.join(self.output_dir, combined_filename)

                        # Guarantee uniqueness even if two groups share a label.
                        if combined_filepath in used_paths:
                            combined_filepath = os.path.join(
                                self.output_dir, f"combined_{combined_timestamp}_{safe}_{chapter_num:02d}.{ext}"
                            )
                        used_paths.add(combined_filepath)

                        # Save Final (User Format, Resampled)
                        self.save_audio(combined_audio_numpy, combined_filepath, format=output_format, target_sample_rate=sample_rate)
                        logger.info(f"Combined audio saved: {combined_filepath}")
                        combined_filepaths.append({
                            "title": group["label"],
                            "path": combined_filepath,
                            "chunk_indices": group["chunk_indices"],
                        })
                    except Exception as save_err:
                        save_failures += 1
                        logger.exception(
                            f"Failed to save chapter {chapter_num} "
                            f"('{group['label']}'): {save_err}. Continuing with remaining chapters."
                        )
                    finally:
                        self._encode_progress_cb = None

                if combine_progress_callback:
                    combine_progress_callback("Done", 100)

                # Only treat the run as failed if NOTHING could be saved.
                if save_failures and not combined_filepaths:
                    raise RuntimeError(f"All {save_failures} chapter file(s) failed to save.")

        except Exception as e:
            logger.exception(f"Synthesis failed: {e}")
            raise

        logger.info("Synthesis complete.")
        return synthesis_result_list, combined_filepaths

    def save_audio(self, audio_data_numpy: np.ndarray, filepath: str, format: str ='WAV', target_sample_rate: int = DEFAULT_SAMPLERATE):
        """Saves audio, resampling if needed, and enforcing PCM_16 for WAV compatibility."""
        
        current_rate = DEFAULT_SAMPLERATE # Kokoro native is 24000
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            target_format_upper = format.upper()

            # --- 1. Resample if needed (Requires librosa) ---
            if target_sample_rate != current_rate:
                # Use librosa to change sample rate (e.g. 24k -> 16k)
                audio_data_numpy = librosa.resample(audio_data_numpy, orig_sr=current_rate, target_sr=target_sample_rate)
            
            # --- 2. Format Conversion (Float32 -> Float32) ---
            if not np.issubdtype(audio_data_numpy.dtype, np.floating):
                 audio_float = audio_data_numpy.astype(np.float32)
            else:
                 audio_float = audio_data_numpy.astype(np.float32)

            # Mono check
            if audio_float.ndim > 1: 
                audio_float_mono = np.mean(audio_float, axis=1)
            else: 
                audio_float_mono = audio_float

            # Clip
            audio_clipped = np.clip(audio_float_mono, -1.0, 1.0)

            # --- 3. Save ---
            if target_format_upper == 'MP3':
                 audio_data_int16 = (audio_clipped * 32767).astype(np.int16)
                 # Pipe raw PCM straight to ffmpeg. pydub's export() first serializes
                 # the whole clip to an in-memory WAV, whose RIFF header stores the data
                 # size in a 32-bit field (~4 GB cap) -> long audiobooks overflow it with
                 # "struct.error: argument out of range". Raw PCM has no size header, so
                 # this avoids the limit entirely.
                 self._encode_mp3_streaming(audio_data_int16, filepath, target_sample_rate)

            elif target_format_upper == 'WAV':
                 # CRITICAL FIX: Use subtype='PCM_16' for UI compatibility
                 sf.write(filepath, audio_clipped, samplerate=target_sample_rate, format='WAV', subtype='PCM_16')
                 # WAV is written in a single blocking call (no incremental progress),
                 # so report this chapter as fully encoded for the compilation meter.
                 cb = getattr(self, "_encode_progress_cb", None)
                 if cb:
                     try:
                         cb(1.0)
                     except Exception as cb_err:
                         logger.debug(f"Progress callback failed: {cb_err}")

            else:
                raise ValueError(f"Unsupported audio format: {format}")

        except Exception as e:
            logger.exception(f"Error saving audio: {e}")
            raise

    def _encode_mp3_streaming(self, pcm_int16: np.ndarray, filepath: str, sample_rate: int):
        """Encode mono int16 PCM to MP3 with ffmpeg.

        Streams the PCM into ffmpeg's stdin from a background thread while reading
        ffmpeg's ``-progress`` output on stdout, so long audiobooks can report a
        live percentage via ``self._encode_progress_cb`` (set per chapter by
        ``synthesize``). stdin/stdout/stderr are each drained on their own thread
        to avoid pipe-buffer deadlocks for large inputs.
        """
        total_seconds = (len(pcm_int16) / float(sample_rate)) if sample_rate else 0.0
        cb = getattr(self, "_encode_progress_cb", None)

        proc = subprocess.Popen(
            ["ffmpeg", "-y",
             "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
             "-i", "pipe:0",
             "-b:a", "192k",
             "-progress", "pipe:1", "-nostats", "-loglevel", "error",
             filepath],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        # Write directly from the array's buffer in chunks instead of materialising
        # one giant bytes() copy — a 100h audiobook would otherwise double its
        # (already large) memory footprint just to hand it to ffmpeg.
        pcm_view = memoryview(np.ascontiguousarray(pcm_int16)).cast("B")
        write_chunk = 1 << 20  # 1 MiB
        # Bounded: we only ever use the tail of stderr for the error message.
        stderr_chunks = deque(maxlen=200)

        def _writer():
            try:
                for off in range(0, len(pcm_view), write_chunk):
                    proc.stdin.write(pcm_view[off:off + write_chunk])
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        def _drain_stderr():
            try:
                for line in proc.stderr:
                    stderr_chunks.append(line)
            except (ValueError, OSError):
                pass

        writer_t = threading.Thread(target=_writer, daemon=True)
        stderr_t = threading.Thread(target=_drain_stderr, daemon=True)
        writer_t.start()
        stderr_t.start()

        try:
            for raw_line in proc.stdout:
                if not cb or total_seconds <= 0:
                    continue
                line = raw_line.decode(errors="ignore").strip()
                # ffmpeg emits both out_time_us and (historically misnamed)
                # out_time_ms in MICROSECONDS; either works divided by 1e6.
                if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                    try:
                        micros = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    frac = max(0.0, min(micros / 1_000_000.0 / total_seconds, 0.999))
                    try:
                        cb(frac)
                    except Exception as cb_err:
                        logger.debug(f"Progress callback failed: {cb_err}")
                elif line == "progress=end":
                    try:
                        cb(1.0)
                    except Exception as cb_err:
                        logger.debug(f"Progress callback failed: {cb_err}")
        finally:
            try:
                proc.stdout.close()
            except OSError:
                pass

        proc.wait()
        writer_t.join(timeout=5)
        stderr_t.join(timeout=5)
        if writer_t.is_alive() or stderr_t.is_alive():
            logger.warning("ffmpeg I/O threads did not finish within timeout.")

        if proc.returncode != 0:
            err = b"".join(stderr_chunks).decode(errors="ignore")[-500:]
            raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {err}")

    def list_available_voices(self):
        return list_available_voices()