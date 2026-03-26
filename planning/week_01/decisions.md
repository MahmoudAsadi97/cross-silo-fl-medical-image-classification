# Week 1 Final Decisions

## Locked Core Choices
- Dataset: Fed-ISIC2019
- Model: ResNet-18
- Baselines: Centralized, Local-only
- Federated algorithms: FedAvg, FedProx, SCAFFOLD
- Privacy method: DP-SGD on selected best FL method
- Primary device: Local PC with RTX 2070
- Optional systems extension: Raspberry Pi client
- Optional infrastructure extension: Azure for artifact/log support

## Initial Experimental Defaults
- Number of clients: 6
- Participation fraction: 1.0
- Communication rounds: 50
- Local epochs: 1
- Batch size: 32
- Learning rate: 0.001
- Seed: 42
- Image size: 224
