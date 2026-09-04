This file defines the rules and operating principles for AI coding agents working in the **Matrix** repository.

Matrix is a five-subsystem UAV video-to-3D geospatial reconstruction system developed for **Smart India Hackathon 2026 (SIH26158)**.

Agents must prioritize **system integrity, clear subsystem boundaries, reproducibility, and integration reliability** over unnecessary complexity.

---

# 1. Repository Context

Matrix is organized into five primary subsystems:

| ID     | Subsystem                    | Responsibility                                     |
| ------ | ---------------------------- | -------------------------------------------------- |
| **S1** | Visual Perception            | Transform UAV video into usable visual information |
| **S2** | Localization & Sensor Fusion | Estimate camera position, trajectory, and pose     |
| **S3** | 3D Reconstruction            | Generate the 3D representation                     |
| **S4** | Georeferencing & Validation  | Georeference and validate the reconstruction       |
| **S5** | Application & Deployment     | Orchestrate, deploy, and visualize the system      |

The canonical system architecture is documented in:

```text
docs/architecture/system.md
```

**This document is authoritative for system-level boundaries and interfaces.**

---

# 2. Core Rule: Architecture Must Stay Current

## 🚨 Mandatory

**After every architectural change — major OR minor — update:**

```text
docs/architecture/system-architecture.md
```

Do not postpone architecture updates until the end of a task, sprint, or milestone.

If an agent changes anything that affects how components fit together, the architecture document must be updated **in the same change/PR**.

This includes seemingly small changes.

### Examples

Update the architecture when changing:

* A subsystem responsibility
* A subsystem boundary
* An input or output
* An interface
* A data format
* A data contract
* A coordinate convention
* A dependency between subsystems
* A processing stage
* A deployment component
* A pipeline sequence
* Error handling across boundaries
* System-level configuration
* Service communication
* Storage architecture
* API contracts
* Authentication/security boundaries
* Major technology choices
* Removal or addition of a system component

### If uncertain

**Update `system-architecture.md`.**

It is better to document a minor architectural change than to allow documentation drift.

---

# 3. Architecture vs Implementation

Agents must distinguish between **system architecture** and **implementation details**.

### Architecture belongs in

```text
docs/architecture/system-architecture.md
```

Examples:

* Components
* Subsystems
* Interfaces
* Data flow
* System dependencies
* Deployment topology
* Cross-system constraints
* System-level decisions

### Implementation belongs in subsystem documentation/code

Examples:

* Algorithms
* Internal classes
* Helper functions
* Model architectures
* Optimization experiments
* Internal data structures
* Temporary implementation details

Do not pollute the system architecture with implementation details that do not affect system-level behavior.

---

# 4. Before Making Changes

Before modifying code, an agent should:

1. Read this `AGENTS.md`.
2. Read the relevant subsystem documentation.
3. Read:

```text
docs/architecture/system-architecture.md
```

4. Read the interface contracts:

```text
docs/architecture/contracts/
```

5. Inspect the existing implementation.
6. Identify upstream and downstream dependencies.
7. Determine whether the proposed change affects an interface.
8. Determine whether the architecture document needs updating.

Do not immediately start coding based only on the task description.

### Architectural change documentation

If the proposed change alters how Matrix components interact (subsystem boundaries, data flow, interfaces, or coordinate conventions), **`docs/architecture/system-architecture.md` must be updated in the same change**. See [23. Architecture Update Standard](#23-architecture-update-standard) and the [🚨 Non-Negotiable Rule](#-non-negotiable-rule) at the end of this document.

---

# 5. Subsystem Boundaries

Each subsystem has one primary responsibility.

Agents must avoid **responsibility leakage**.

### S1 — Visual Perception

Owns:

* Video/image processing
* Frame extraction
* Keyframe selection
* Visual information generation
* Visual metadata

Does **not** own final localization or geographic alignment.

### S2 — Localization & Sensor Fusion

Owns:

* Position estimation
* Trajectory estimation
* Camera pose
* Sensor fusion

Does **not** own 3D reconstruction.

### S3 — 3D Reconstruction

Owns:

* 3D reconstruction
* Point clouds
* Mesh generation
* Reconstruction representation

Does **not** own final georeferencing.

### S4 — Georeferencing & Validation

Owns:

* Geographic alignment
* Coordinate transformation
* Spatial validation
* Reconstruction quality evaluation

Does **not** own the user-facing application.

### S5 — Application & Deployment

Owns:

* User interface
* API/application orchestration
* Processing lifecycle
* Visualization
* Deployment

Does **not** own internal algorithms of S1–S4.

---

# 6. Interface-First Development

When modifying or creating a subsystem, think in terms of:

```text
INPUT
  ↓
CONTRACT
  ↓
SUBSYSTEM
  ↓
CONTRACT
  ↓
OUTPUT
```

Every external interface should clearly define:

* Input
* Output
* Data format
* Required metadata
* Preconditions
* Guarantees
* Failure conditions
* Version

Avoid undocumented assumptions.

### Never do this

```python
result = downstream_process(data["some_field"])
```

if `some_field` is not part of the documented contract.

### Prefer

A documented, validated interface with explicit expectations.

---

# 7. Validate at Boundaries

Subsystems should validate data received from other subsystems.

Do not assume upstream data is always valid.

At minimum, consider:

* Required fields
* Data types
* Units
* Coordinate systems
* Timestamp consistency
* Shape/dimensions
* File validity
* Missing values
* Version compatibility

Invalid input should produce a clear error.

Do not silently fabricate missing information.

---

# 8. Coordinate Systems Are Critical

Matrix operates with multiple possible spatial representations:

```text
Image / Camera Coordinates
          ↓
Local Reconstruction Coordinates
          ↓
Geographic / World Coordinates
```

Agents must never assume these coordinate systems are interchangeable.

Any code dealing with spatial information should make the following explicit where relevant:

* Coordinate system
* Units
* Origin
* Orientation
* Reference frame
* Transformation

If a coordinate convention changes, **update `system-architecture.md` immediately.**

---

# 9. Changes to Interfaces

Interface changes are higher-risk than internal implementation changes.

Before changing an interface:

1. Identify all consumers.
2. Identify all producers.
3. Check the architecture documentation.
4. Update the interface.
5. Update affected implementations.
6. Update tests.
7. Update documentation.
8. Update `docs/architecture/system-architecture.md`.
9. Clearly describe the breaking/non-breaking nature of the change.

Never silently change a shared contract.

---

# 10. Dependencies

Before introducing a dependency, ask:

* Is it necessary?
* Does the project already have an equivalent?
* Does it increase deployment complexity?
* Does it require GPU/CUDA/system packages?
* Does it affect licensing?
* Does it create a runtime dependency?
* Does it increase build time?
* Does it affect reproducibility?

Prefer simple, well-supported dependencies.

Avoid adding libraries merely because they are fashionable.

---

# 11. Python Rules

For Python code:

* Use clear module boundaries.
* Prefer type hints.
* Keep functions focused.
* Avoid unnecessary global state.
* Handle exceptions deliberately.
* Do not silently swallow errors.
* Keep configuration separate from implementation.
* Avoid hard-coded machine-specific paths.
* Do not commit credentials.
* Keep generated data outside source control unless explicitly required.

Virtual environments must not be committed.

---

# 12. TypeScript / JavaScript Rules

For TypeScript/JavaScript:

* Prefer TypeScript for new application code.
* Keep types explicit at system boundaries.
* Avoid unnecessary `any`.
* Validate external data.
* Keep API contracts synchronized with backend contracts.
* Avoid embedding secrets in frontend code.
* Keep generated build artifacts out of Git.
* Prefer reusable components over duplicated logic.

---

# 13. Data & Large Files

Do not commit large generated datasets, UAV videos, point clouds, meshes, model weights, or other generated artifacts unless explicitly required.

Typical local/generated files include:

```text
data/raw/
data/processed/
data/output/
models/
checkpoints/
```

The repository `.gitignore` defines the current policy.

If a large artifact is required for reproducibility, discuss its storage strategy before committing it.

---

# 14. Testing

Every meaningful code change should include appropriate validation.

Depending on the change, this may include:

* Unit tests
* Integration tests
* Interface/contract tests
* End-to-end tests
* Static checks
* Type checks
* Manual validation

For subsystem changes, prioritize testing both:

```text
Internal correctness
        +
Interface correctness
```

A subsystem that works independently but breaks its consumer is not considered correct.

---

# 15. Integration Testing

For changes crossing subsystem boundaries, test the actual integration.

Examples:

```text
S1 → S2
S2 → S3
S3 → S4
S4 → S5
```

Do not rely exclusively on mocked interfaces when a real integration path is available.

For important pipeline changes, run an end-to-end representative workflow.

---

# 16. Error Handling

Errors should be explicit and observable.

Matrix recognizes several system-level error categories:

```text
Input Error
     ↓
Processing Error
     ↓
Quality Warning
     ↓
Integration Error
     ↓
System Error
```

Agents should:

* Preserve useful error context.
* Avoid swallowing exceptions.
* Provide actionable messages.
* Fail early when required input is invalid.
* Avoid silently producing misleading output.

If a change introduces a new system-level failure mode, document it.

---

# 17. Logging & Observability

Agents should make it possible to answer:

> **Where did the pipeline fail?**

Useful information includes:

* Current subsystem
* Processing state
* Duration
* Errors
* Warnings
* Output availability
* Pipeline/version information

Do not expose secrets or sensitive data in logs.

---

# 18. Configuration

Configuration should be explicit.

Prefer:

```text
environment variables
configuration files
CLI arguments
```

over hard-coded values.

Never commit:

* API keys
* Passwords
* Tokens
* Private credentials
* Production secrets

Use `.env.example` for documenting required environment variables.

---

# 19. Git Workflow

Use feature branches.

```text
main
 │
 ├── feature/perception
 ├── feature/localization
 ├── feature/reconstruction
 ├── feature/georeferencing
 └── feature/application
```

Do not directly push experimental or incomplete work to `main`.

Preferred flow:

```text
Branch
  ↓
Implement
  ↓
Test
  ↓
Update documentation
  ↓
Update architecture if applicable
  ↓
Pull Request
  ↓
Review
  ↓
Merge
```

---

# 20. Commit Messages

Use concise, meaningful commit messages.

Examples:

```text
feat: add video frame extraction
feat: add pose interface
fix: handle missing telemetry timestamps
fix: validate reconstruction metadata
refactor: simplify point cloud pipeline
test: add localization contract tests
docs: update reconstruction architecture
chore: update dependencies
```

Avoid messages such as:

```text
update
changes
stuff
final
final final
please work
```

The Git history should communicate engineering intent.

---

# 21. Pull Requests

A pull request should explain:

### What changed?

Short summary of the implementation.

### Why?

Problem or requirement being addressed.

### Architectural impact?

State explicitly:

* No architectural impact
* Minor architectural change
* Major architectural change

### Interface impact?

State whether any subsystem contract changed.

### Testing

Explain what was tested.

### Documentation

Confirm whether relevant documentation was updated.

### Architecture

**If there is any architectural impact, `docs/architecture/system-architecture.md` must be updated in the same PR.**

---

# 22. Documentation Synchronization

Documentation must evolve with implementation.

When changing a subsystem:

```text
Code Change
    │
    ├── Tests
    ├── Subsystem Documentation
    └── Architecture Update
```

Do not intentionally leave known architecture/documentation drift.

### Mandatory architecture update trigger

If a reasonable engineer reading the diff could ask:

> "Does this change alter how Matrix components interact?"

then update:

```text
docs/architecture/system-architecture.md
```

---

# 23. Architecture Update Standard

When updating `system-architecture.md`, update only the affected sections unless the change requires broader restructuring.

Preserve:

* Existing architectural intent
* Clear subsystem boundaries
* Interface definitions
* Data-flow diagrams
* Ownership model
* System constraints

Do not rewrite unrelated architecture simply because you are editing the file.

Architecture documentation should describe the **current system**, not historical states.

If an important decision needs historical context, record it separately under:

```text
docs/decisions/
```

---

# 24. Major vs Minor Architectural Changes

### Minor

Examples:

* New metadata field
* Small interface clarification
* New validation requirement
* Additional processing status
* Non-breaking data format extension

Still update:

```text
docs/architecture/system-architecture.md
```

### Major

Examples:

* New subsystem
* Removed subsystem
* Changed subsystem responsibility
* Changed pipeline sequence
* New communication mechanism
* Major deployment change
* Breaking interface change
* New system-wide dependency
* Major coordinate-system change

For major changes, also consider creating an Architecture Decision Record under:

```text
docs/decisions/
```

---

# 25. Do Not Over-Engineer

Matrix is a hackathon project.

The goal is a **working, credible, demonstrable system**.

Avoid introducing complexity without measurable value.

Before adding:

* Microservices
* Message queues
* Distributed infrastructure
* Complex orchestration
* New databases
* Multiple abstraction layers
* Heavy frameworks

ask:

> **Does this materially improve the demonstrated system?**

If not, prefer the simpler solution.

---

# 26. Research vs Production Code

Matrix contains both research and engineering work.

Keep experimental work isolated where practical.

```text
Research
   ↓
Experiment
   ↓
Evaluation
   ↓
Validated approach
   ↓
Production integration
```

Do not treat an experimental result as production-ready merely because it produces an output.

Record important experiments under:

```text
docs/experiments/
```

---

# 27. Agent Autonomy

Agents may independently:

* Inspect the repository.
* Modify implementation files.
* Add tests.
* Refactor internal code.
* Improve documentation.
* Fix bugs.
* Update subsystem documentation.
* Update architecture documentation when required.

Agents should **not** independently make major product or architectural decisions when those decisions affect multiple subsystem owners without clearly documenting the impact.

When a change crosses subsystem boundaries, surface the architectural implications.

---

# 28. Preserve Existing Work

Agents must avoid unnecessary destructive changes.

Before modifying files:

* Inspect existing implementation.
* Understand current behavior.
* Preserve working functionality.
* Avoid broad rewrites unless necessary.
* Do not overwrite teammate work without understanding it.

Prefer small, reviewable changes.

---

# 29. Definition of Done for Agent Tasks

An agent task is complete only when applicable:

* [ ] Implementation is complete.
* [ ] Existing functionality remains intact.
* [ ] Tests pass.
* [ ] New behavior is tested.
* [ ] Relevant subsystem documentation is updated.
* [ ] Interfaces remain valid.
* [ ] Architecture is reviewed.
* [ ] `docs/architecture/system-architecture.md` is updated if the change affects architecture.
* [ ] No secrets or unnecessary generated files are committed.
* [ ] The resulting change is understandable to another engineer.

---

# 30. Final Agent Checklist

Before considering a task complete, ask:

```text
□ Did I understand the existing architecture?
□ Did I stay within the correct subsystem boundary?
□ Did I preserve existing interfaces?
□ If I changed an interface, did I update every consumer?
□ Did I add appropriate tests?
□ Did I update subsystem documentation?
□ Did I update architecture.md?
□ Did I introduce unnecessary complexity?
□ Did I create any undocumented dependency?
□ Did I leave generated/secrets/local files out of Git?
□ Can another engineer understand what I changed?
```

## 🚨 Non-Negotiable Rule

> **Every major or minor architectural change MUST be reflected in `docs/architecture/system-architecture.md` in the same change that introduces it.**

Architecture is not documentation that gets written once.

**The architecture document is a living representation of the system.**

If the implementation and architecture disagree, the repository is considered inconsistent until the discrepancy is resolved.

# Reuse Before Reinventing

## Principle

**Prefer composition and reuse over implementing functionality from scratch.**

Before writing non-trivial functionality, the agent MUST determine whether the requirement can be satisfied by an existing implementation.

The preferred order is:

1. **Existing project functionality**
2. **Standard library**
3. **Existing dependencies**
4. **Mature third-party package**
5. **New implementation**

Do not reinvent functionality that is already solved well by a reliable, maintained dependency.

## Mandatory Reuse Check

Before creating a new utility, helper, abstraction, subsystem, or significant piece of functionality, evaluate:

* Does the repository already provide this?
* Does the language standard library provide this?
* Does an existing project dependency provide this?
* Is there a mature third-party package that provides this?
* Would using an existing solution introduce unacceptable complexity, security risk, performance problems, or incompatibility?
* If implementing from scratch, what specifically justifies doing so?

For non-trivial functionality, explicitly record the conclusion in the implementation plan or task notes.

### Decision Rule

If an established package satisfies the requirements with reasonable compatibility and maintenance characteristics:

> **Use the package rather than implementing equivalent functionality yourself.**

Only implement from scratch when there is a concrete reason not to reuse an existing solution.

## Examples

Prefer established packages or existing abstractions for functionality such as:

* Retry and exponential backoff
* Schema validation
* JWT/OAuth handling
* PDF parsing
* HTML parsing
* Database migrations
* Scheduling
* Caching
* Queues
* Fuzzy matching
* Serialization formats
* Cryptographic primitives
* Complex date/time operations

Do **not** add a dependency merely to avoid writing a few trivial lines of code. Small, obvious functionality should remain simple when the standard library or a tiny local implementation is clearly superior.

## Why This Matters

Unnecessary custom implementations increase:

* Codebase size
* Maintenance burden
* Bug surface
* Testing requirements
* Cognitive load
* Dependency on undocumented internal behavior
* Difficulty for future engineers and agents to understand the system

Agents should optimize for **the smallest robust solution**, not the largest amount of newly generated code.

### Core Heuristic

> **Before asking "How do I implement this?", ask "Who has already implemented this well?"**

The goal is not to minimize lines of code at all costs. The goal is to minimize **unnecessary complexity and duplicated functionality** while preserving correctness, reliability, security, and maintainability.
