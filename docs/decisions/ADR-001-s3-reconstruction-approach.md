# ADR-001: S3 3D Reconstruction Approach

**Status:** Accepted  
**Date:** 2026-09-01  
**Author:** S3 Subsystem Lead  
**Deciders:** Matrix Architecture Team  

---

## 1. Context & Problem Statement

Subsystem **S3 (3D Reconstruction)** is tasked with transforming visual observations from S1 and camera poses/trajectories from S2 into a high-density, geometrically accurate 3D point cloud and spatial metadata in a single-pass UAV workflow.

Given the hackathon constraints (robustness, reproducibility, zero external heavyweight binary dependencies, edge CPU/GPU execution), we need a reconstruction approach that:
1. Operates reliably given calibrated pinhole camera models and $4\times 4$ poses.
2. Scales across multi-view feature tracks ($N \ge 2$ observations per 3D landmark).
3. Enforces strict geometric validity (positive depth / cheirality check, ray angle / parallax filtering).
4. Accurately calculates per-point and global reprojection errors for S4 validation.
5. Provides a clear path to dense surface reconstruction and meshing.

---

## 2. Considered Approaches

### Approach A: Multi-View SVD Direct Linear Transformation (DLT) with Robust Filtering (Selected)
* **Mechanism:** Formulates multi-view ray intersections as an overdetermined linear system $A X = 0$ for each feature track across $N$ camera views, solved via Singular Value Decomposition (SVD). Followed by cheirality verification ($Z_{cam} > 0$), baseline parallax thresholding ($\theta \ge 2.0^\circ$), and reprojection error pruning.
* **Pros:**
  * Highly performant, pure NumPy implementation with zero external C++ binary dependency.
  * Deterministic and mathematically rigorous.
  * Scales naturally from 2-view to $N$-view tracks.
  * Direct computation of per-point covariance and reprojection error.
* **Cons:** Point density depends on the density of 2D feature tracks extracted from visual observations.

### Approach B: Dense Depth Map Estimation & Volumetric Fusion (MVS)
* **Mechanism:** Computes pairwise or multi-view depth maps per frame via patch matching or plane sweep, then fuses depth maps into a truncated signed distance field (TSDF) or voxel grid.
* **Pros:** Generates dense continuous surface point clouds.
* **Cons:** High computational and memory cost; sensitive to low-texture terrain without GPU acceleration.

### Approach C: Neural Radiance Fields (NeRF) / 3D Gaussian Splatting (3DGS)
* **Mechanism:** Optimizes continuous volumetric radiance field or 3D Gaussian primitives from camera viewpoints.
* **Pros:** Photorealistic novel view synthesis.
* **Cons:** Requires substantial training time per scene, GPU requirement (CUDA), high memory overhead, and does not natively produce georeferenceable metric point clouds without post-extraction.

---

## 3. Decision

We choose **Approach A (Multi-View SVD DLT with Geometric Filtering & Ray Verification)** as the primary reconstruction engine for Subsystem S3.

The architecture is structured with a modular `ReconstructionEngineBase` interface so that dense MVS or learned depth fusion can be added as alternative or complementary backend engines without modifying S2 or S4 interfaces.

---

## 4. Consequences & Guarantees

* **Positive:** S3 is completely self-contained, reproducible on any standard Python environment without GPU or proprietary libraries, and executes in real-time.
* **Guarantees:**
  * Local coordinate system: strictly `S3_LOCAL` in meters.
  * All points in the output point cloud satisfy positive depth ($Z_{cam} > 0$) in all contributing cameras.
  * Points with mean reprojection error above threshold (default $3.0\text{ px}$) are rejected.
  * Outputs standard binary/ASCII `scene.ply` and `metadata.json` compatible with S4's `ReconstructionInput`.

