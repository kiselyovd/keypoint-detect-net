"""Lightning module wrappers."""
from __future__ import annotations

import lightning as L
import torch
from torch import nn, optim

# Ultralytics YOLO manages its own training loop; no Lightning wrapper needed.
