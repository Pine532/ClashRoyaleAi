"""Opponent tracking: what a viewer can know about the other side.

`opponent_tracker` is pure Python plus numpy (no torch, no engine import), so the
training env and `perception/` run the same code on their respective event
streams. `engine_feed` is the simulator-side adapter and may import the engine.
"""
