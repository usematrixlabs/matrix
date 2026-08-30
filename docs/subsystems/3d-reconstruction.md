# S3 — 3D Reconstruction

## Responsibility

S3 generates the **3D representation of the observed environment** using the visual observations and camera information provided by S2.

S3 may perform:

* Multi-view reconstruction
* Structure-from-motion-related processing
* Depth estimation
* Point-cloud generation
* Surface reconstruction
* Mesh generation
* Texture generation where applicable
* Reconstruction quality assessment

### S3 does not own

* Primary video ingestion
* Primary frame extraction
* Primary localization
* GPS/IMU fusion
* Final geographic alignment
* Final geographic validation
* User interface
* Application orchestration

Internal reconstruction methods may perform local optimization or refinement where required. Such implementation details do not change the subsystem boundary.

### S3 answers:

> **What does the observed environment look like in 3D?**