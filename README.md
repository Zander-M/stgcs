# ST-GCS
This repository implements the Graphs of Space-Time Convex Sets (ST-GCS) from the following papers:

- Branch master: Jingtao Tang, Zining Mao, Lufan Yang, and Hang Ma. "Search-Based Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets." [[paper]](https://arxiv.org/abs/2607.00444), [[project]](https://sites.google.com/view/stgcs)
- Branch iros: Jingtao Tang, Zining Mao, Lufan Yang, and Hang Ma. "Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning."  [[paper]](https://arxiv.org/abs/2503.00583), [[project]](https://sites.google.com/view/stgcs)


## Installation
- Install the Python project from the repository root with `pip install -e .`
- [Mosek](https://www.mosek.com/) solver should be installed, which is used in the [Drake](https://drake.mit.edu/) library for the GCS program solving. 
- [Optional] You can also use Gurobi solver for Drake, however, building Drake from source is required. See detailed installation guidance [here](https://drake.mit.edu/installation.html).


## Quick Start

To be updated

## File Structure

To be updated

## BibTex:
```bibtex
@misc{tang2026searchbasedspatiotemporalmultirobotmotion,
  title={Search-Based Spatiotemporal and Multi-Robot Motion Planning on Graphs of Space-Time Convex Sets},
  author={Jingtao Tang and Zining Mao and Lufan Yang and Hang Ma},
  year={2026},
  eprint={2607.00444},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2607.00444}
}
```
IROS Version:
```bibtex
@misc{tang2025spacetimegraphsconvexsets,
      title={Space-Time Graphs of Convex Sets for Multi-Robot Motion Planning}, 
      author={Jingtao Tang and Zining Mao and Lufan Yang and Hang Ma},
      year={2025},
      eprint={2503.00583},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.00583}, 
}
```

## License
ST-GCS is released under the GPL version 3. See LICENSE.txt for further details.

