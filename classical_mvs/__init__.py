"""Classical multi-view stereo reconstruction pipeline.

Reads calibrated RGB views (EasyMocap format from Motion-Capture) and
produces a watertight triangle mesh via plane-sweep MVS + TSDF fusion.
"""
