# ArgoCast

[![Code DOI](https://zenodo.org/badge/1379753522.svg)](https://doi.org/10.5281/zenodo.23422343)
[![Data DOI](https://img.shields.io/badge/Data%20DOI-10.5281%2Fzenodo.23423759-blue)](https://doi.org/10.5281/zenodo.23423759)
[![Release](https://img.shields.io/badge/release-v2.0.0-blue)](https://github.com/VighneshKunjir/ArgoCast/releases/tag/v2.0.0)
[![Code license: MIT](https://img.shields.io/badge/Code%20license-MIT-yellow.svg)](LICENSE)
[![Active branch: gai-build](https://img.shields.io/badge/active%20branch-gai--build-brightgreen)](https://github.com/VighneshKunjir/ArgoCast/tree/gai-build)

> **This branch is not the current project.**
> `main` only holds an early frontend prototype from the first stage of development.
> The complete, maintained version of ArgoCast (backend, trained model, benchmark, research code and documentation) is on the **[`gai-build`](https://github.com/VighneshKunjir/ArgoCast/tree/gai-build)** branch.

**[Go to the gai-build branch](https://github.com/VighneshKunjir/ArgoCast/tree/gai-build)**

---

## What ArgoCast is

ArgoCast forecasts the next temperature and salinity profile of an Argo float in the Arabian Sea (16 levels, 5 to 1000 dbar, about 10 days ahead) from its three most recent profiles. The forecasting model is a bidirectional LSTM trained with a TEOS-10 static-stability penalty. Each forecast comes with calibrated uncertainty bands, Integrated Gradients explanations, links to similar observed Argo profiles and a grounded question-answering panel.

## Where to find things

| What | Where |
|---|---|
| Full source code, setup guide and results | [`gai-build` branch](https://github.com/VighneshKunjir/ArgoCast/tree/gai-build) |
| Archived code release (v2.0.0) | https://doi.org/10.5281/zenodo.23422344 |
| Processed data, data splits and benchmark results | https://doi.org/10.5281/zenodo.23423759 |
| Releases | https://github.com/VighneshKunjir/ArgoCast/releases |

## Getting the current version

Clone the `gai-build` branch directly:

```
git clone -b gai-build https://github.com/VighneshKunjir/ArgoCast.git
cd ArgoCast
```

If you already cloned the repository:

```
git fetch origin
git checkout gai-build
```

Then follow the Quick start in that branch's README.

## What is on this branch

| | `main` (this branch) | `gai-build` |
|---|---|---|
| Status | Early prototype, not maintained | Current version |
| Frontend | First React layout | Full app: forecast, chat, sea map, XAI and benchmarks tabs |
| Backend and model | Not included (placeholder data) | FastAPI backend with the trained PI-BiLSTM model |
| Data | 4-float sample | 36,164 profiles from 321 floats (2001-2026) |
| Benchmark | None | 13 models, two test splits, three seeds |

The files on this branch are kept only as a record of the project's history. Please do not use them for new work.

## Authors

Vighnesh Walmik Kunjir and Shakti Kinger, Department of Computer Science and Engineering, MIT World Peace University (MIT-WPU), Pune, India.

## License

Code: MIT License, see [LICENSE](LICENSE). Processed data on Zenodo: CC BY 4.0.
