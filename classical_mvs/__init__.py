"""Classical multi-view stereo reconstruction pipeline.

Reads calibrated RGB views (EasyMocap format from Motion-Capture), **requires**
foreground masks under ``<data>/masks/<cam>/``, and produces a triangle mesh
via plane-sweep MVS + TSDF fusion.
"""
