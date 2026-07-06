"""Protected validation for the CIFAR-10 speedrun — vendored from bench/validate.py.

Kept byte-for-byte semantically identical so runner numbers sit on the same axis
as the bench/ numbers: 6-view TTA (mirror + 1px translate) computed runner-side
from the model's `eval_forward(x) -> logits`, on the held-out graded split.

The test set splits into two fixed halves:
  - FEEDBACK split (first 2,000 test images): accuracy returned to the loop for
    steering only (trajectory mode).
  - GRADED split (remaining 8,000): the pass/fail accuracy and reported metric.

Validation uses `no_grad`, NOT `inference_mode`: inference tensors poison
torch.compile / CUDA-graph training afterwards (aborted captures corrupt RNG).
See audit finding #17 — preserve this.
"""

import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))
EVAL_BATCH = 2000
FEEDBACK_N = 2000  # test[0:2000] -> feedback split; test[2000:] -> graded split

_TEST_CACHE: dict = {}  # keyed by absolute data_path (audit finding #15)


def _load_test(data_path):
    key = os.path.abspath(data_path)
    if key in _TEST_CACHE:
        return _TEST_CACHE[key]
    data = torch.load(os.path.join(data_path, "test.pt"), map_location="cuda", weights_only=True)
    images = (data["images"].half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
    images = T.Normalize(CIFAR_MEAN, CIFAR_STD)(images)
    labels = data["labels"].cuda()
    _TEST_CACHE[key] = (images, labels)
    return _TEST_CACHE[key]


def _tta_logits(model, images):
    def net(x):
        return model.eval_forward(x).float()

    def mirror(x):
        return 0.5 * net(x) + 0.5 * net(x.flip(-1))

    out = []
    for x in images.split(EVAL_BATCH):
        logits = mirror(x)
        px = F.pad(x, (1,) * 4, "reflect")
        translate = torch.stack([mirror(px[:, :, 0:32, 0:32]), mirror(px[:, :, 2:34, 2:34])]).mean(0)
        out.append(0.5 * logits + 0.5 * translate)
    return torch.cat(out)


def run_validation(model, data_path, rank=0, world_size=1, **kwargs):
    """TTA top-1 accuracy on both splits:
    {'feedback': acc on test[0:2000], 'graded': acc on test[2000:]}."""
    if not hasattr(model, "eval_forward"):
        raise RuntimeError(
            "Model must implement eval_forward(x) -> logits. The runner computes "
            "accuracy from logits so the model cannot return a fabricated metric."
        )
    images, labels = _load_test(data_path)
    model.eval()
    with torch.no_grad():
        logits = _tta_logits(model, images)
    correct = (logits.argmax(1) == labels).float()
    model.train()
    return {
        "feedback": correct[:FEEDBACK_N].mean().item(),
        "graded": correct[FEEDBACK_N:].mean().item(),
    }
