"""Light-Aware GS - relight a decomposed splat inside LichtFeld Studio.

Treat the loaded splat's DC color as recovered ALBEDO, derive a per-Gaussian normal from each disk's
shortest axis, and re-shade with a user-movable light (albedo * (ambient + intensity * relu(n.l))).
"Capture albedo" snapshots the current colors as the material; move the light to relight live.
"""

import lichtfeld as lf
from .panels.relight_panel import RelightPanel

_classes = [RelightPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("light_aware plugin loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("light_aware plugin unloaded")
