# ADR-002: Independent Subsystems & Pipeline-Owned Integration

**Status:** Adopted
**Date:** 2026-09-02
**Deciders:** Matrix Architecture Team
**Related:** `docs/architecture/system.md`, `docs/architecture/system-architecture.md`, `docs/architecture/contracts/`

---

## 1. Context

Matrix is composed of five subsystems:

* **S1 — Visual Perception**
* **S2 — Localization & Sensor Fusion**
* **S3 — 3D Reconstruction**
* **S4 — Georeferencing & Validation**
* **S5 — Application & Deployment**

The subsystems are designed as **independent modules with explicit interfaces/contracts**.

Previously, subsystem status was sometimes evaluated based on whether the complete end-to-end pipeline successfully produced the expected final artifact. This conflates two separate concerns:

1. Whether an individual subsystem is correctly implemented.
2. Whether the pipeline correctly obtains, transforms, and passes data between subsystems.

This distinction is critical for Matrix because a subsystem may be fully implemented while the pipeline is currently unable to provide it with valid inputs.

---

## 2. Decision

### Subsystems are independent modules. The pipeline owns integration.

Each **subsystem** is responsible for:

* Implementing its own algorithms and business logic.
* Defining and honoring its input/output contract.
* Validating the data it receives against its contract.
* Producing its own outputs, metrics, diagnostics, and status.
* Handling invalid, insufficient, or degraded inputs according to its contract.

The **pipeline** is responsible for connecting these independent modules. The pipeline owns:

* Execution order.
* Invoking subsystems.
* Obtaining outputs from upstream subsystems.
* Adapting data between subsystem contracts.
* Passing inputs to downstream subsystems.
* Managing artifact locations.
* Propagating status and failures.
* End-to-end orchestration.
* Cross-subsystem validation.

The pipeline is therefore **not part of the internal implementation of any subsystem**.

---

## 3. Responsibility Boundary

```
┌─────────────────┐
│ S1              │
│ Visual          │
│ Perception      │
└────────┬────────┘
         │
         │ S1 contract
         ▼
┌─────────────────────────────────────┐
│             PIPELINE                │
│                                     │
│  • orchestration                    │
│  • adaptation                       │
│  • data movement                    │
│  • execution order                  │
│  • artifact management              │
│  • status propagation               │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ S2              │
│ Localization &  │
│ Sensor Fusion   │
└────────┬────────┘
         │
         │ S2 contract
         ▼
       PIPELINE
         │
         ▼
┌─────────────────┐
│ S3              │
│ 3D              │
│ Reconstruction  │
└────────┬────────┘
         │
         │ S3 contract
         ▼
       PIPELINE
         │
         ▼
┌─────────────────┐
│ S4              │
│ Georeferencing  │
│ & Validation    │
└────────┬────────┘
         │
         ▼
       PIPELINE
         │
         ▼
┌─────────────────┐
│ S5              │
│ Application &   │
│ Deployment      │
└─────────────────┘
```

---

## 4. Consequences for Subsystem Status

Subsystem maturity must be assessed based on **the subsystem itself**, not merely on the current state of the integrated pipeline.

For example — **S3**:

If S3 can:

```
valid S3 input
      ↓
reconstruction
      ↓
point cloud + metrics
```

and its implementation passes its own tests, then S3 can be considered implemented even if the current pipeline fails to provide it with valid camera poses.

The following is **not automatically an S3 defect**:

```
S2
 ↓
invalid/incomplete data
 ↓
Pipeline
 ↓
S3 receives unusable input
 ↓
0 points
```

That is primarily a **pipeline integration issue**, provided S3 correctly handles the invalid input according to its contract.

---

## 5. Data Ownership

Subsystems do **not** obtain data directly from other subsystems. Instead:

```
S1 output
   ↓
Pipeline obtains S1 output
   ↓
Pipeline adapts it
   ↓
S2 input
```

Likewise:

```
S2 output
   ↓
Pipeline obtains S2 output
   ↓
Pipeline adapts it
   ↓
S3 input
```

and:

```
S3 output
   ↓
Pipeline obtains S3 output
   ↓
Pipeline adapts it
   ↓
S4 input
```

This prevents hidden coupling between subsystem implementations.

---

## 6. What a Subsystem Must NOT Do

A subsystem should not:

* Reach into another subsystem's internal code.
* Read another subsystem's private artifacts directly.
* Assume how upstream data was generated.
* Perform another subsystem's algorithmic responsibilities.
* Implement pipeline orchestration.
* Make assumptions about the source of its input beyond its contract.

For example, **S3 should not care whether camera poses came from** GPS, visual odometry, EKF, COLMAP, an external system, or synthetic data. S3 only needs a valid **S3 input contract**. Similarly, S4 should not care how S3 generated its point cloud. It consumes the agreed S4 input.

---

## 7. Contract-First Integration

Every subsystem boundary must have an explicit contract (`docs/architecture/contracts/`).

A contract should define:

**Input:**

* Required fields
* Optional fields
* Data types
* Units
* Coordinate frames
* Valid ranges
* Quality requirements

**Output:**

* Artifacts
* Schemas
* Metrics
* Status
* Diagnostics
* Failure/degradation semantics

The pipeline is responsible for converting the output of one contract into the input expected by the next contract.

---

## 8. Status Classification

When diagnosing Matrix, use three separate statuses:

| # | Status | Question |
|---|--------|----------|
| 1 | **Module status** | Is the subsystem itself implemented correctly? |
| 2 | **Integration status** | Is the pipeline correctly connecting the subsystem to the rest of Matrix? |
| 3 | **End-to-end status** | Does the complete system successfully process real benchmark data? |

These must not be conflated. All three can be simultaneously true:

```
S3 module:             ✅ Implemented
S2 → S3 integration:   ❌ Invalid/incomplete
End-to-end pipeline:   ❌ Fails
```

---

## 9. Application to Current Matrix State

Under this architecture:

| Component | Correct assessment |
|-----------|--------------------|
| **S1** | Mostly implemented; calibration strategy for uncalibrated inputs remains |
| **S2** | Core components exist, but visual localization execution/fusion remains incomplete |
| **S3** | Core reconstruction implementation substantially complete; real-data validation and interface formalization remain |
| **S4** | Core georeferencing infrastructure exists; real control-point/CRS/Helmert workflow remains |
| **S5** | Largely incomplete; currently primarily a bundling stub |
| **Pipeline** | Significant integration work remains, including data adaptation, S2 invocation, status propagation, and multi-flight execution |

Importantly, the current **zero-point S3 benchmark result does not by itself imply that S3's reconstruction algorithm is incomplete**. The pipeline currently fails to provide S3 with the valid camera geometry and observations required by its contract.

---

## 10. Engineering Rule — Trace to the Contract Boundary

When a downstream subsystem fails, trace the failure to the **contract boundary** before assigning ownership:

```
Did the upstream subsystem produce valid output?
        │
        ├── NO → upstream subsystem issue
        │
        └── YES
             ↓
Did the pipeline correctly adapt/pass it?
        │
        ├── NO → pipeline integration issue
        │
        └── YES
             ↓
Did downstream subsystem correctly process it?
        │
        ├── NO → downstream subsystem issue
        │
        └── YES → investigate subsequent boundary
```

This prevents incorrectly modifying a subsystem to compensate for failures elsewhere.

---

## 11. Principle

> **Subsystems own computation. The pipeline owns composition. Contracts own boundaries.**

This principle governs Matrix architecture, task ownership, debugging, testing, and progress reporting going forward.

---

## 12. Consequences

* **Positive:** Clear ownership for bugs and features; prevents blame misattribution; enables parallel subsystem development; contracts become the source of truth for integration testing.
* **Negative:** Requires disciplined contract maintenance; pipeline adapters must be kept in sync with contract versions.
* **Neutral:** Existing code must be audited to remove any direct cross-subsystem reads that bypass the pipeline. See `docs/architecture/system-architecture.md` §5 (Pipeline Orchestrator) for mandated adapter locations.
