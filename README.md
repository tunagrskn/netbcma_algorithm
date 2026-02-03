# NetBMCA - Network-Aware Best Master Clock Algorithm

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-2.6+-orange.svg)

**A topology-aware extension of IEEE 802.1AS/1588 BMCA for optimal grandmaster selection in TSN networks**

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Algorithm](#-algorithm) •
[Results](#-results) •
[Citation](#-citation)

</div>

---

## Overview

**NetBMCA** (Network-aware Best Master Clock Algorithm) extends the standard IEEE 802.1AS/1588 BMCA by incorporating network topology metrics into the grandmaster (GM) selection process. While traditional BMCA selects the GM based solely on clock quality, NetBMCA considers both clock quality and network centrality to minimize synchronization path lengths across the network.

### The Problem

In Time-Sensitive Networking (TSN), the standard BMCA may select a high-quality GPS clock located at the network edge as the grandmaster. This forces all nodes to synchronize with a remote GM, resulting in:

- Longer synchronization paths
- Increased round-trip delays
- Higher jitter accumulation

### The Solution

NetBMCA evaluates candidate grandmasters using a weighted scoring function that combines:

- **Clock Quality**: Priority, class, accuracy (IEEE 1588 standard)
- **Network Centrality**: Betweenness, closeness, eigenvector centrality

This approach selects a **topologically central** node as GM, reducing average synchronization delays across the network.

---

## Features

- **Standard BMCA Implementation** - Full IEEE 802.1AS-2020 compliant BMCA
- **Network-Aware Extension** - Centrality-based GM selection
- **Adaptive Weights** - Automatically tunes weights based on network characteristics
- **TSN Network Generator** - Synthetic automotive and industrial topologies
- **Validation Framework** - Comprehensive testing on diverse network types
- **Visualization Tools** - Publication-ready figures and analysis

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/tgirisken/netbcma_algorithm.git
cd netbcma_algorithm

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `networkx` | ≥2.6.0 | Graph operations & centrality |
| `numpy` | ≥1.20.0 | Numerical computations |
| `pandas` | ≥1.3.0 | Data analysis |
| `scipy` | ≥1.7.0 | Statistical analysis |
| `matplotlib` | ≥3.4.0 | Visualization |
| `seaborn` | ≥0.11.0 | Enhanced plots |

---

## Quick Start

### Demo Mode (3 Networks)

```bash
python3 main.py
```

### Full Validation (20 Networks)

```bash
python3 main.py --validate
```

### Detailed Analysis

```bash
python3 main.py --detailed
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--validate`, `-v` | Run full validation on 20 networks |
| `--no-figures` | Skip figure generation |
| `--detailed`, `-d` | Verbose step-by-step analysis |

---

## Algorithm

### Standard BMCA (IEEE 802.1AS)

The standard Best Master Clock Algorithm compares clocks using the following priority order:

1. **Priority1** (0-255, lower is better)
2. **Clock Class** (6=GPS, 248=Default)
3. **Clock Accuracy** (25ns to 100ms)
4. **Offset Scaled Log Variance** (Allan variance)
5. **Priority2** (tiebreaker)
6. **Node ID** (final tiebreaker)

### NetBMCA Extension

NetBMCA introduces a **weighted scoring function**:

```
Score(n) = w₁·Priority(n) + w₂·Class(n) + w₃·Betweenness(n) 
         + w₄·Closeness(n) + w₅·Eigenvector(n)
```

**Default Weights (Medium Networks):**

| Weight | Value | Metric |
|--------|-------|--------|
| w₁ | 0.30 | Priority |
| w₂ | 0.25 | Clock Class |
| w₃ | 0.25 | Betweenness Centrality |
| w₄ | 0.15 | Closeness Centrality |
| w₅ | 0.05 | Eigenvector Centrality |

### Adaptive Weight Tuning

Weights are automatically adjusted based on network characteristics:

| Network Size | Focus |
|--------------|-------|
| Small (<20 nodes) | Clock quality prioritized |
| Medium (20-100 nodes) | Balanced approach |
| Large (>100 nodes) | Topology prioritized |

---

## Results

### Validation Networks

NetBMCA was validated on 20 diverse network topologies:

| Type | Count | Description |
|------|-------|-------------|
| **Scale-Free** | 5 | Barabási-Albert model |
| **Small-World** | 5 | Watts-Strogatz model |
| **Geometric** | 5 | Random geometric graphs |
| **Powerlaw Cluster** | 3 | Powerlaw cluster graphs |
| **Control** | 2 | Regular & ring (symmetric) |

### Key Findings

- **Average Improvement**: 15-25% reduction in round-trip delay
- **Best Results**: Geometric and scale-free networks with asymmetric topology
- **Strong Correlation**: Improvement correlates with network centrality asymmetry

---

## Project Structure

```
netbcma_algorithm/
├── main.py                    # Main entry point & CLI
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── README.md                  # This file
│
├── algorithms/
│   ├── __init__.py
│   ├── standard_bmca.py       # IEEE 802.1AS BMCA implementation
│   └── net_bmca.py            # Network-aware BMCA extension
│
├── utils/
│   ├── __init__.py
│   ├── network_utils.py       # Network analysis utilities
│   └── tsn_generator.py       # TSN topology generator
│
├── results/
│   └── figures/               # Generated analysis figures
│
└── doc/
    └── tuna_girisken_netbcma_paper.tex  # LaTeX paper
```

---

## API Usage

### Basic Example

```python
import networkx as nx
from algorithms import StandardBMCA, NetBMCA, BMCADataSet, ClockQuality, ClockClass

# Create a network
G = nx.barabasi_albert_graph(50, 3)

# Define clock properties
clocks = {
    0: BMCADataSet(node_id=0, priority1=128, 
                   clock_quality=ClockQuality(clock_class=ClockClass.GPS)),
    5: BMCADataSet(node_id=5, priority1=128,
                   clock_quality=ClockQuality(clock_class=ClockClass.GPS)),
    # ... other nodes default to slave clocks
}

# Run algorithms
std_bmca = StandardBMCA(G, clocks)
net_bmca = NetBMCA(G, clocks)

gm_standard = std_bmca.run()
gm_netbmca = net_bmca.run()

print(f"Standard BMCA GM: Node {gm_standard}")
print(f"NetBMCA GM: Node {gm_netbmca}")
```

### TSN Network Generation

```python
from utils.tsn_generator import TSNTopologyGenerator

# Automotive network (gateway + domain controllers + ECUs)
G_auto, desc = TSNTopologyGenerator.create_automotive_network(num_ecus=20)

# Industrial network (PLC + IO devices in ring topology)
G_ind, desc = TSNTopologyGenerator.create_industrial_network(num_devices=30, ring=True)
```

---

## Visualization

Run validation with figure generation:

```bash
python3 main.py --validate
```

Generated figures are saved to `results/figures/`:

- **Improvement Analysis**: Bar chart of delay improvements by network type
- **Correlation Plot**: Centrality asymmetry vs. improvement correlation

---

## Academic Context

This project was developed as part of the **Complex Networks** course at **Ege University**.

### Research Contributions

1. **Novel Algorithm**: First topology-aware BMCA extension for TSN
2. **Adaptive Weighting**: Self-tuning based on network characteristics
3. **Comprehensive Validation**: Tested on diverse synthetic networks
4. **Open Implementation**: Fully reproducible Python implementation

---

## Citation

If you use NetBMCA in your research, please cite:

```bibtex
@software{girisken2026netbmca,
  author = {Girişken, Tuna},
  title = {NetBMCA: Network-Aware Best Master Clock Algorithm},
  year = {2026},
  institution = {Ege University},
  url = {https://github.com/tgirisken/netbcma_algorithm}
}
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Author

**Tuna Girişken**  
Ege University

---

<div align="center">

Made for Time-Sensitive Networking research

</div>
