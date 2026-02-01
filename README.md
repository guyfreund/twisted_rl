<h1 align="center">
    TWISTED-RL: Hierarchical Skilled Agents for Knot-Tying without Human Demonstrations
</h1>

<h2 align="center">
    Guy Freund<sup>1</sup>, Tom Jurgenson<sup>2</sup>, Matan Sudry<sup>2</sup> and Erez Karpas<sup>2,3</sup>
</h2>
<h4 align="center">
   <sup>1</sup>Reichman University, 
   <sup>2</sup>Technion — Israel Institute of Technology,
   <sup>3</sup>The University of Texas at Austin
</h4>
<h3 align="center">
   Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026.
</h3>
<p align="center">
  <a href="https://sites.google.com/view/twisted-rl">
    <img src="https://img.shields.io/badge/Project_Page-🔗-blue?style=for-the-badge" alt="Project Page">
  </a>
  <a href="YOUR_PAPER_URL_HERE">
    <img src="https://img.shields.io/badge/Paper-📄-red?style=for-the-badge" alt="Paper">
  </a>
</p>

<p align="left">
Official implementation of <strong>TWISTED-RL</strong>, a hierarchical framework for robotic knot-tying that combines a high-level topological planner with low-level reinforcement learning policies. The framework enables robots to learn complex knot-tying skills without human demonstrations.
</p>

---

[//]: # (![TWISTED-RL Inference]&#40;resources/TWISTED-RL.png&#41;)
[//]: # (![TWISTED-RL Training]&#40;resources/TWISTED-RL-train.png&#41;)

[//]: # (<img src="images/TWISTED-RL.png.png" alt="TWISTED-RL Inference" width="600"/>)
[//]: # (<img src="images/TWISTED-RL-train.png.png" alt="TWISTED-RL Training" width="600"/>)

## 🔧 Installation 

This project uses Conda for environment management.

```bash
# Clone the repository
git clone https://github.com/guyfreund/twisted_rl.git
cd twisted_rl

# Create and activate conda environment
conda env create -f server_environment_39_18_04_2025.yml
conda activate twisted_rl
```

---

## 💡 Best Practices
1. Run all scripts with the root dir `.../twisted_rl` as the working directory.

2. It is recommended to run the problem set file to create all relevant problems:
   ```bash
   python exploration/mdp/graph/problem_set.py
   ```
---

## 📈 Training

TWISTED-RL is trained hierarchically across levels of increasing complexity.

To train the TWISTED-RL-C variant, run the following:

#### Crossing Number 0 (G0)

```bash
python exploration/rl/cleanrl_scripts/sac_algorithm.py \
    --problem G0
```

#### Crossing Number 1 (G1):

```bash
python exploration/rl/cleanrl_scripts/sac_algorithm.py \
    --problem G1 \
    --replay_buffer_files_path PATH_TO_G0
```

#### Crossing Number 2 (G2):

```bash
python exploration/rl/cleanrl_scripts/sac_algorithm.py \
    --problem G2 \
    --replay_buffer_files_path PATH_TO_G1
```

### ⚙️ Configuration
Edit the following to adjust training configuration:
- `exploration/rl/config/sac.yml` for training parameters.
- `exploration/rl/environment/exploration_env.yaml` for environment parameters.

---

## 📊 Inference & Evaluation

To run evaluation of the full TWISTED-RL system:

```bash
python system_flow/evaluation_automation.py \
    --ablation C
```

### ⚙️ Configuration
Edit `system_flow/config/twisted_evaluation.yml` to adjust evaluation configuration.
```yaml
LOW_LEVEL:
  RL:
    agents:
      G0: path/to/G0.pt
      G1: path/to/G1.pt
      G2: path/to/G2.pt

EVALUATION:
  states_type: 3-eval  # or easy, medium, hard, 4-eval
```

---

## 📁 Project Structure

```plaintext
twisted_rl/
├── exploration/
│   ├── goal_selector/               # Goal selection strategies
│   ├── initial_state_selector/      # Initial state selection strategies
│   ├── mdp/                         # MDP definitions
│   ├── preprocessing/               # Data processing
│   ├── reachable_configurations/    # Reachability graph
│   ├── rl/
│   │   ├── cleanrl_scripts/         # Training scripts
│   │   ├── config/                  # Training configs
│   │   ├── environment/             # Custom environments
│   │   ├── replay_buffer/           # Replay buffers
│   │   └── test_scripts/            # Evaluation scripts
│   └── utils/                       # Helper functions
├── mujoco_infra/                    # MuJoCo simulation wrappers and assets
├── system_flow/
│   ├── config/                      # System-level configs
│   ├── high_level_class/            # High-level planners
│   ├── low_level_class/             # Low-level planners
│   └── metrics/                     # Evaluation metrics metadata
```

---

## 📖 Citation
If you find **TWISTED-RL** useful in your research, please consider citing our work:
```bibtex
@inproceedings{freund2026twistedrl,
  title     = {TWISTED-RL: Hierarchical Skilled Agents for Knot-Tying without Human Demonstrations},
  author    = {Freund, Guy and Jurgenson, Tom and Sudry, Matan and Karpas, Erez},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
  note      = {To appear}
}
```
---

## ⚖️ License
This repository is licensed under the MIT License. See the [LICENSE](https://github.com/guyfreund/twisted_rl/tree/master/LICENSE) file for details.