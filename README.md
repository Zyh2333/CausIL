# CausIL: Causal Graph for Instance Level Microservice Data

This is the official repository corresponding to the paper titled "CausIL: Causal Graph for Instance Level Microservice Data"  accepted at the Proeedings of The Web Conference 2023 (WWW '23), Austin, Texas, USA.

**Please cite our paper in any published work that uses any of these resources.**


## Abstract

AI-based monitoring has become crucial for cloud-based services due to its scale. A common approach to AI-based monitoring is to detect causal relationships among service components and build a causal graph. Availability of domain information makes cloud systems even better suited for such causal detection approaches. In modern cloud systems, however, auto-scalers dynamically change the number of microservice instances, and a load-balancer manages the load on each instance. This poses a challenge for off-the-shelf causal structure detection techniques as they neither incorporate the system architectural domain information nor provide a way to model distributed compute across varying numbers of service instances. To address this, we develop CausIL, which detects a causal structure among service metrics by considering compute distributed across dynamic instances and incorporating domain knowledge derived from system architecture. Towards the application in cloud systems, CausIL estimates a causal graph using instance-specific variations in performance metrics, modeling multiple instances of a service as independent, conditional on system assumptions. Simulation study shows the efficacy of CausIL over baselines by improving graph estimation accuracy by ~25% as measured by Structural Hamming Distance whereas the real-world dataset demonstrates CausIL's applicability in deployment settings.