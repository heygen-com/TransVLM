"""TransVLM inference: shot transition detection with optical flow injected at the input stage.

Two entry points sit beside this package rather than inside it, because they are commands rather
than library code:

* ``infer_video.py`` -- one video in, merged whole-video transitions out.
* ``infer_clips.py`` -- pre-cut clips in, per-clip predictions out. This is the route the
  published numbers were produced with, and it pairs with ``evaluation/``.

Nothing here imports a backend at module scope. ``vllm``, ``sglang`` and the HuggingFace stack are
each pulled in only when the corresponding engine is constructed, so ``--help`` stays fast and a
venv that has only one backend installed still works.
"""
