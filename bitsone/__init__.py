"""The BitSOne system-one package.

Layout:

    bitsone/paths.py    filesystem locations, resolved in one place
    bitsone/model.py    load the base; check its quantisation mode
    bitsone/corpus.py   the six corpus files, the renderer, the key-token rule
    bitsone/pointer.py  the head, its loss, the batch builders
    bitsone/serve.py    render -> key tokens -> head, one entry point
    bitsone/train_head.py / train_joint.py   the two trainers
    bitsone/eval.py     read-once evaluation with receipts

Nothing here may import `checks/`: the gates depend on this package, never the reverse, so
`import bitsone` stays torch-free and CI can use it.
"""

from __future__ import annotations
