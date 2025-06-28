# Shor-Algorithm

# Quantum Algorithm Simulations Using Python

## 🧪 Simulation of Shor's, Deutsch-Jozsa, and Grover Algorithms using QuTiP and Matplotlib

This repository contains Python-based simulations of three foundational quantum algorithms:
- **Shor's Algorithm** (for integer factorization)
- **Deutsch-Jozsa Algorithm** (for function analysis)
- **Grover's Search Algorithm** (for unstructured database search)

All simulations use:
- [QuTiP](https://qutip.org/)  – for quantum state manipulation
- [Matplotlib](https://matplotlib.org/)  – for Bloch sphere visualization
- [FFmpeg](https://ffmpeg.org/)  – for animation saving

The goal is to make complex quantum concepts accessible through **computational experimentation**, especially for non-physicists.

---

## 📋 Features

- Menu-driven interface for easy interaction
- Animated **Bloch spheres** showing quantum state evolution
- Probability bar charts at each step
- Key frames saved as `.png` files for academic use
- Full implementation of all three algorithms
- Designed by a **Chemical Engineer** exploring quantum mechanics computationally

---

## 🎥 Animations & Visualizations

Each algorithm includes:
- An MP4 animation (`shor_animation.mp4`, `deutsch_jozsa_2qubit.mp4`, `grover_animation.mp4`)
- Saved key frames like:
  - `shor_frame_0.png`
  - `shor_frame_299.png`
  - `shor_frame_899.png`
- Probability distribution graphs

These visuals help explain abstract quantum concepts in an intuitive way.

---

## 🛠️ Requirements

To run this project locally or share on GitHub, install:

```bash
pip install qutip matplotlib numpy sympy pillow
