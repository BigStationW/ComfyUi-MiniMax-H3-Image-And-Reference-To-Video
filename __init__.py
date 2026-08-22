import math
import logging
import inspect
from functools import wraps


import nodes
import node_helpers
from comfy_api.latest import ComfyExtension, io

# Re-use the exact helpers/constants the built-in H3 nodes use.
from comfy_extras.nodes_minimax_h3 import (
    _empty_av_latent,
    _resize,
    MiniMaxH3ReferenceToVideo,
    adapt_canvas,
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    FPS,
)


def _encode_ref_audio(audio_vae, audio):
    """Use ComfyUI's current MiniMax H3 reference-audio implementation.

    In current ComfyUI this helper is a static method on
    MiniMaxH3ReferenceToVideo rather than a module-level function.
    """
    return MiniMaxH3ReferenceToVideo._encode_ref_audio(audio_vae, audio)


_H3_PAYLOAD_PATCH_MARKER = "_minimax_h3_keyframe_ref_merge_patch_v2"
_H3_LAYOUT_PATCH_MARKER = "_minimax_h3_keyframe_ref_layout_patch_v1"
_H3_TARGET_ANCHOR_KEY = "_minimax_h3_combined_target_anchor"
_LOG = logging.getLogger("ComfyUI.MiniMaxH3ImageAndReference")


def _is_our_combined_keyframes(keyframes):
    return bool(keyframes) and any(
        isinstance(kf, dict) and kf.get(_H3_TARGET_ANCHOR_KEY, False)
        for kf in keyframes
    )


def _ensure_h3_keyframe_ref_merge():
    """Let this pack's keyframes and Ref2VA refs coexist on older ComfyUI.

    Older MiniMaxH3.extra_conds builds keyframe visual latents first, then the
    refs branch overwrites that list. PackedLayout still contains both sets of
    fixed rows, so sampling either crashes or assigns the wrong latent to the
    wrong rows. Rebuild the list in layout order: keyframes first, refs second.

    This wrapper is deliberately scoped to keyframes produced by this pack.
    """
    try:
        import comfy.model_base as model_base
    except Exception as e:
        _LOG.warning("MiniMax H3 payload patch: could not import comfy.model_base: %s", e)
        return False

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("MiniMax H3 payload patch: MiniMaxH3.extra_conds not found.")
        return False

    current = getattr(cls, "extra_conds")
    if getattr(current, _H3_PAYLOAD_PATCH_MARKER, False):
        return True

    @wraps(current)
    def _patched_extra_conds(self, **kwargs):
        out = current(self, **kwargs)

        keyframes = kwargs.get("minimax_keyframes", None)
        refs = kwargs.get("minimax_refs", None)
        if not refs or not _is_our_combined_keyframes(keyframes):
            return out

        cond = out.get("minimax_payload", None) if isinstance(out, dict) else None
        payload = getattr(cond, "cond", None) if cond is not None else None
        if not isinstance(payload, dict):
            _LOG.warning(
                "MiniMax H3 payload patch: could not access minimax_payload; "
                "combined keyframes+refs may still fail."
            )
            return out

        payload["cond_video_latents"] = (
            [kf["latent"] for kf in keyframes if isinstance(kf, dict) and "latent" in kf]
            + [ref["latent"] for ref in refs if isinstance(ref, dict) and "latent" in ref]
        )
        payload["cond_audio_latents"] = [
            ref["audio_latent"]
            for ref in refs
            if isinstance(ref, dict) and ref.get("audio_latent") is not None
        ]

        frame_count = kwargs.get("minimax_frame_count", None)
        if frame_count is not None:
            payload["frame_count"] = frame_count

        return out

    setattr(_patched_extra_conds, _H3_PAYLOAD_PATCH_MARKER, True)
    cls.extra_conds = _patched_extra_conds
    _LOG.info("MiniMax H3 payload patch: enabled keyframe+reference latent merge.")
    return True


def _ensure_h3_keyframe_ref_layout():
    """Keep first/last keyframes anchored to the TARGET clip when refs exist.

    Old PackedLayout versions calculate first/last keyframe time coordinates
    from ``text_len`` while Ref2VA references advance the target video's origin.
    The result is a first-frame guide sitting in reference-time rather than at
    target frame zero. New ComfyUI versions fixed this natively by computing
    keyframe coordinates from the reference-adjusted cursor; those versions are
    detected by the absence of the old ``frame_count`` constructor parameter.
    """
    try:
        import comfy.ldm.minimax.model as mm
    except Exception as e:
        _LOG.warning("MiniMax H3 layout patch: could not import minimax model: %s", e)
        return False

    cls = getattr(mm, "PackedLayout", None)
    if cls is None:
        _LOG.warning("MiniMax H3 layout patch: PackedLayout not found.")
        return False

    current = getattr(cls, "__init__", None)
    if current is None:
        return False
    if getattr(current, _H3_LAYOUT_PATCH_MARKER, False):
        return True

    # ComfyUI's native guide/ref coexistence update removed frame_count and
    # already uses: cond_t = target_origin + FRAME_RESCALE * frame_index.
    try:
        params = inspect.signature(current).parameters
    except (TypeError, ValueError):
        params = {}
    if "frame_count" not in params:
        _LOG.info("MiniMax H3 layout patch: native target-relative keyframe layout detected; no patch needed.")
        return True

    @wraps(current)
    def _patched_layout_init(
        self, text_len, latent_t, latent_h, latent_w, audio_t,
        keyframes=None, refs=None, frame_count=None,
    ):
        current(
            self, text_len, latent_t, latent_h, latent_w, audio_t,
            keyframes=keyframes, refs=refs, frame_count=frame_count,
        )

        if not refs or not _is_our_combined_keyframes(keyframes):
            return

        segments = getattr(self, "segments", None)
        position_ids = getattr(self, "position_ids", None)
        if not segments or position_ids is None:
            raise RuntimeError(
                "MiniMax H3 combined conditioning: PackedLayout no longer exposes "
                "segments/position_ids; cannot safely align first_frame to target frame 0."
            )

        # Old H3 stores finalized segments as (start_row, end_row, kind).
        target = None
        cond_segments = []
        for seg in segments:
            if not isinstance(seg, (tuple, list)) or len(seg) != 3:
                continue
            a, b, kind = seg
            if kind == "cond":
                cond_segments.append((int(a), int(b)))
            elif kind == "video":
                target = (int(a), int(b))

        if target is None:
            raise RuntimeError(
                "MiniMax H3 combined conditioning: target video segment was not found "
                "in PackedLayout; refusing to guess keyframe positions."
            )
        if len(cond_segments) != len(keyframes):
            raise RuntimeError(
                "MiniMax H3 combined conditioning: keyframe/cond segment count mismatch "
                f"({len(keyframes)} keyframes vs {len(cond_segments)} cond segments)."
            )

        target_a, target_b = target
        if target_b <= target_a:
            raise RuntimeError("MiniMax H3 combined conditioning: target video segment is empty.")

        target_origin = float(position_ids[target_a, 0])
        offset = target_origin - float(text_len)
        if abs(offset) < 1e-12:
            return

        # Stock old-core first/last keyframe arithmetic is correct relative to
        # text_len. References simply move the target origin forward, therefore
        # adding the same target offset to each cond span preserves first/last
        # semantics while moving them onto the generated clip's timeline.
        for a, b in cond_segments:
            position_ids[a:b, 0] = position_ids[a:b, 0] + offset

        _LOG.debug(
            "MiniMax H3 layout patch: shifted %d keyframe cond segment(s) by %.6f "
            "to target origin %.6f.", len(cond_segments), offset, target_origin,
        )

    setattr(_patched_layout_init, _H3_LAYOUT_PATCH_MARKER, True)
    cls.__init__ = _patched_layout_init
    _LOG.info("MiniMax H3 layout patch: enabled target-relative keyframes with references.")
    return True


class MiniMaxH3ImageAndReferenceToVideo(io.ComfyNode):
    """
    Union of MiniMaxH3ImageToVideo (fl2va keyframes) and
    MiniMaxH3ReferenceToVideo (ref2va references) inputs, in one node.

    - first_frame / last_frame behave exactly like MiniMaxH3ImageToVideo
      (they occupy real frame positions in the output and are added to the
      conditioning as `minimax_keyframes`).
    - ref_images / ref_videos / ref_video_audios / ref_audios behave exactly
      like MiniMaxH3ReferenceToVideo (identity/motion/voice steering, no fixed
      frame position, added to the conditioning as `minimax_refs`).
    - Both mechanisms can be used together or independently; either group of
      inputs may be left empty.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageAndReferenceToVideo",
            display_name="MiniMax H3 Image + Reference to Video",
            description=(
                "Combined fl2va + ref2va conditioning for MiniMax H3. "
                "Supports first/last keyframes AND <Picture i> / <Video k> / "
                "<Audio j> references in the same conditioning payload."
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s; trained range is ~124-362, longer is untested)",
                ),

                # --- from MiniMaxH3ImageToVideo (fl2va) ---
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),

                # --- from MiniMaxH3ReferenceToVideo (ref2va) ---
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, "
                            "keeping aspect) to the generation's pixel area; 'max' uses the "
                            "reference pipeline's 2048px short edge for best identity "
                            "fidelity. Reference tokens ride through every sampling step, "
                            "so 'max' can be several times slower.",
                ),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Reference image (downscaled to 2048 short edge if "
                                    "larger, never upscaled)",
                        ),
                        names=[f"ref_image_{i}" for i in range(1, 11)], # 1 through 10
                        min=0, 
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input(
                            "ref_video", tooltip="Reference video frames at 24 fps (2-15s)"
                        ),
                        names=[f"ref_video_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack of the same-numbered reference video",
                        ),
                        names=[f"ref_video_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        names=[f"ref_audio_{i}" for i in range(1, 5)], # 1 through 4
                        min=0,
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(
        cls, clip, vae, audio_vae, prompt, width, height, length,
        first_frame=None, last_frame=None,
        ref_image_size="match",
        ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None,
    ) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        # ---------- fl2va: first/last keyframes ----------
        images = []
        keyframes = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img, _H3_TARGET_ANCHOR_KEY: True})
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img, _H3_TARGET_ANCHOR_KEY: True})

        # ---------- ref2va: images / videos / audio references ----------
        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # for the DiT payload, same order

        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({
                "type": "video", "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_idx))],
            })
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent,
            })

        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        # ---------- tokenize + encode (both keyframe images and refs at once) ----------
        tokens = clip.tokenize(prompt, images=images, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        payload = {}
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            payload["minimax_keyframes"] = keyframes
            payload["minimax_frame_count"] = frame_count
        if ref_blocks:
            payload["minimax_refs"] = ref_blocks
        if payload:
            cond = node_helpers.conditioning_set_values(cond, payload)

        return io.NodeOutput(cond, latent)

_ensure_h3_keyframe_ref_merge()
_ensure_h3_keyframe_ref_layout()


class MiniMaxH3ImageAndReferenceExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ImageAndReferenceToVideo]

async def comfy_entrypoint() -> ComfyExtension:
    _ensure_h3_keyframe_ref_merge()
    _ensure_h3_keyframe_ref_layout()
    return MiniMaxH3ImageAndReferenceExtension()
