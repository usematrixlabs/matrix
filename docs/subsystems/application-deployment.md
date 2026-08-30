# S5 — Application & Deployment

## Responsibility

S5 is the **system-facing layer**.

S5 owns:

* User input
* File upload
* Pipeline initiation
* Pipeline orchestration
* Job/process management
* Processing status
* Error/status presentation
* Result delivery
* 3D visualization
* Application deployment
* Runtime integration

S5 connects the processing pipeline to the user.

### S5 does not own

* Visual perception algorithms
* Localization algorithms
* Sensor-fusion algorithms
* Reconstruction algorithms
* Georeferencing algorithms
* Validation methodology

S5 **orchestrates** these capabilities; it does not absorb their responsibilities.

### S5 answers:

> **How does a user run Matrix and interact with its output?**