# CFB-FAS: Towards Robust and Secure Cross-Domain Face Anti-Spoofing via Feature-Boundary Consistency

A robust face anti-spoofing framework with cross-domain generalization capability, leveraging dynamic clustering and boundary-aware losses to enhance feature consistency.


## Overview
CFB-FAS addresses the challenge of cross-domain generalization in face anti-spoofing (FAS) by focusing on two key aspects:  
- **Feature Consistency**: Using Dynamic Soft K-Means Loss (DSK-Loss) to enforce compact clustering of live samples within the same domain.  
- **Boundary Consistency**: Using Boundary-aware Dynamic Center Loss (BDC-Loss) to sharpen decision boundaries between live and spoof samples.  

The framework achieves state-of-the-art performance on standard cross-domain FAS benchmarks and unseen 3D attack scenarios.


## Key Features
- **Dynamic Clustering**: DSK-Loss adaptively optimizes intra-domain compactness and inter-domain separability for live samples.  
- **Boundary Regularization**: BDC-Loss enhances feature discriminability at decision boundaries via dynamic center contraction and adversarial constraints.  
- **Cross-Domain Robustness**: Evaluated on 5 benchmark datasets, with strong generalization to unseen domains and 3D attacks.  


## Dependencies
- Python 3.8+  
- PyTorch 1.10+  
- torchvision 0.11+  
- OpenCV 4.5+  
- numpy, scipy, tqdm, sklearn  
