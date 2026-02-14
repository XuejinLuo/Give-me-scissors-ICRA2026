<h1 align="center">Give me Scissors: Collision-Free Dual-Arm Surgical Assistive Robot for Instruments Delivery</h1>

<h3 align="center">Zero-shot surgical instrument delivery powered by VLMs and real-time QP collision avoidance.</h3>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg" alt="arXiv"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="Project Homepage"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%8E%AC%20Video-YouTube-red" alt="YouTube Video"></a>
  &nbsp;
  <a href="https://img.shields.io/badge/license-MIT-blue.svg"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  &nbsp;
  <a href="https://img.shields.io/badge/ROS-2-green"><img src="https://img.shields.io/badge/ROS-2-green" alt="ROS2"></a>
</p>

<div align="center">
    <img src="assets/demo.gif" width="80%" alt="Dual-arm robot handing over scissors smoothly while avoiding obstacles">
    <p><em>Our dual-arm robot autonomously plans and delivers surgical instruments while dynamically avoiding collisions in real-time.</em></p>
</div>

## 🔥 Updates

- **[2026-02]** 🔥🔥🔥 We released the official codebase for *Give me Scissors*, including VLM integration and the real-time QP collision avoidance framework!
- **[2026-02]** 🚀 Paper submitted to ICRA 2026. Stay tuned for the arXiv link and full project page!

## 🕶️ Overview

[cite_start]During the perioperative phase, scrub nurses are required to frequently deliver surgical instruments to surgeons, which can lead to physical fatigue and decreased focus[cite: 4]. [cite_start]Traditional robotic scrub nurses rely on predefined pathways, which limits their generalization and poses safety risks in dynamic environments[cite: 6]. 

[cite_start]To address these challenges, we present a collision-free dual-arm surgical assistive robot capable of zero-shot instrument delivery[cite: 7]. 

<div align="center"> 
    <img src="assets/pipeline.png" alt="System Pipeline" style="width=90%;vertical-align:middle">
    <p><em>System Architecture: Multi-modal VLM task planning meets real-time QP reactive collision avoidance.</em></p>
</div>

### ✨ Key Features
- [cite_start]**Zero-Shot VLM Task Planning**: Utilizes Vision Language Models (e.g., GPT-4o) to automatically generate grasping and delivery sub-goals from surgeon's natural language instructions and visual features[cite: 8, 425].
- [cite_start]**Unified QP Collision Avoidance**: A unified Quadratic Programming (QP) framework ensures real-time reactive obstacle avoidance and self-collision avoidance simultaneously during autonomous movement[cite: 10, 108].
- [cite_start]**Markerless Dynamic Perception**: Predicts the minimum distance between robot links and environmental obstacles in real-time without relying on visual markers[cite: 102].
- [cite_start]**High Success Rate**: Achieves an 83.33% success rate in real-world surgical instrument delivery tasks with smooth, collision-free motion[cite: 11].

## 🛠️ Installation

We recommend using [Conda](https://docs.conda.io/en/latest/) to create and manage your Python environment.

```bash
# Clone the repository
git clone [https://github.com/XuejinLuo/Give-me-scissors-ICRA2026.git](https://github.com/XuejinLuo/Give-me-scissors-ICRA2026.git)
cd Give-me-scissors-ICRA2026

# Create the environment directly from the YAML file
conda env create -f requirement.yml

# Activate the environment
conda activate surgery_robot
