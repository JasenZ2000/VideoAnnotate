"""Process-isolated LocateAnything GPU service.

Each configured CUDA device owns one spawned Python process and therefore one
independent copy of LocateAnything's process-global batch runtime.
"""

