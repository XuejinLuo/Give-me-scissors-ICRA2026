没问题！这是为你量身定制的完整版“顶配” README。我已经将你专属的 GitHub 仓库地址、Conda `requirement.yml` 安装方式，以及所有高级排版元素完美融合在了一起。

你可以直接点击代码块右上角的“复制”，然后粘贴到你项目根目录的 `README.md` 文件中。

---

```markdown
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

```

*(Note: Ensure your ROS 2 workspace is properly sourced if you are running the hardware or simulation interfaces.)*

## 🚀 Quick Start

The main entry point is the `run.py` script. You can run different modes by specifying the arguments.

### 1. VLM-Guided Surgical Delivery

Execute the automated instrument delivery pipeline (e.g., picking up tweezers and releasing them to the surgeon):

```bash
python run.py --mode rekep --task surgery_tool

```

*(Tip: Add `--use_cached_query` to bypass the VLM query and use locally cached programs for faster debugging.)*

### 2. Collision Avoidance Sandbox

Test the dual-arm environment and self-collision avoidance features in the testing sandbox:

```bash
python run.py --mode test --enable_self_collision_avoidance --enable_env_collision_avoidance

```

### 3. Custom Natural Language Instructions

Override predefined tasks by passing natural language instructions directly:

```bash
python run.py --mode rekep --instruction "Robot 1 picks up the tweezers. Then Robot 1 holds it above 10 cm. Then Robot 1 releases the tweezers on the desk."

```

## ⚙️ Hardware Setup

The real-world experimental platform for this system consists of:

* 
**Robotic Arms**: 2x Franka Research 3 robotic arms.


* 
**Low-Level Control**: Industrial computers running Ubuntu 22.04 LTS and PREEMPT_RT kernel at 1 kHz.


* 
**Perception**: 3x Intel RealSense D435i RGB-D cameras.


* 
**Middleware**: ROS 2.







