# TrackNetV3 tracking module

Source: [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3), pinned at
`6eda442ada1740573f200f836d93edc9a541ee86`. See `source.json` for upstream file hashes
and modifications. `tracknet.py` retains the original 2D blocks and tracking network;
the 1D blocks and InpaintNet are omitted. Attribution is added and trailing whitespace
removed. The MIT copyright and permission notice is preserved in `LICENSE`.

The upstream license explicitly includes pretrained model checkpoints. The checkpoint
itself is not vendored. `download_tracknet.py` fetches the author-linked archive and
checks the recorded archive and tracking-checkpoint fingerprints. These are hashes of
the artifacts evaluated locally, not a claim of an upstream cryptographic signature.

BadmintonAI uses its own scene-bounded CPU inference adapter in `shuttle_tracknet.py`.
It supports concat-background checkpoints, restores native-image coordinates, and
does not call InpaintNet or interpolate missing observations. Its inference-window
schedule and scene-local median differ from the author's published full pipeline.
Results must be evaluated locally; the upstream paper's reported metrics do not apply
to BadmintonAI's footage or adapter.
