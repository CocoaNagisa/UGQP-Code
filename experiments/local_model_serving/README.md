## Contents

- `workflow_contracts.py`: workflow scenes, prompts, typed-field scoring, and completed-node feedback.
- `proxy_calibration.py`: 36 independent stage-level cases and 216 endpoint calls.
- `endpoint_calibration.py`: separate 24-workflow, 384-call endpoint calibration.
- `build_endpoint_profile.py`: reconstructs `runtime/endpoint_profile.json` from the endpoint-calibration records.
- `routing_comparison.py`: prepares, trains, seals, executes, and analyzes the 36-workflow UGQP-GNN/DTODRL-joint comparison.
- `runtime/`: the simulator modules and small DTODRL initialization checkpoint used for this comparison. It is isolated from the original main-comparison code.
