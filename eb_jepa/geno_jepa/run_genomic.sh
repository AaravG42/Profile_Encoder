#!/bin/bash
# Example script to train JEPA model on genomic data

# Train with ResNet backbone
python -m examples.image_jepa.main \
    --fname examples/image_jepa/cfgs/genomic.yaml

# Train with Vision Transformer backbone
python -m examples.image_jepa.main \
    --fname examples/image_jepa/cfgs/genomic.yaml \
    model.type=vit_s \
    model.patch_size=41

# Train with BCS loss instead of VICReg
python -m examples.image_jepa.main \
    --fname examples/image_jepa/cfgs/genomic.yaml \
    loss.type=bcs \
    loss.lmbd=0.12

# Override config parameters
python -m examples.image_jepa.main \
    --fname examples/image_jepa/cfgs/genomic.yaml \
    optim.epochs=100 \
    data.batch_size=64 \
    optim.lr=0.05

# Multiple seeds for seed averaging
for seed in 1 42 100; do
    python -m examples.image_jepa.main \
        --fname examples/image_jepa/cfgs/genomic.yaml \
        meta.seed=$seed
done
