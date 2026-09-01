"""Matrix end-to-end pipeline orchestrator.

A thin wrapper that invokes each subsystem's main exported package in
sequence, passing outputs from one stage to the next. It does not reach
into any subsystem's internals — it only calls the documented public
entry points.
"""
