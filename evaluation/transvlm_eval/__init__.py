"""The scoring pipeline the published TransVLM numbers were computed with.

Two ways in:

* :func:`score` -- dataset-agnostic. Give it ground truth, predictions and per-video durations as
  three plain dicts keyed by whatever video id you use, and get the 11 time tolerances plus their
  mean. Use this on your own data.
Clip-level predictions can be lifted to video time first with
:func:`transvlm_eval.assembly.lift_clip_predictions`, which takes your clip spans and a callback --
it has no opinion about where the clips came from.

The metric is segment-level, swept over 11 time tolerances from 0.00 s to 0.50 s in 0.05 s steps,
with the mean across them reported as ``ttmean``. Matching is bare temporal overlap, many-to-many,
with no IoU threshold; counts are aggregated micro (summed across videos, then divided) rather
than macro.

**The durations argument is not the container duration.** See :func:`score` -- getting it wrong
changes every number, and it is the one thing about this scorer that is not guessable.

Three primitives are vendored rather than reimplemented, in :mod:`transvlm_eval._vendored`: the
matcher, the NMS merge, and the shot-boundary value types. See that package's docstring for why.
"""

from transvlm_eval.pipeline import score

__all__ = ['score']
