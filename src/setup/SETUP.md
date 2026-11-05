# TrafficFlowBench competition environment (Windows, Python 3.13.7)

The environment is the stock Python 3.13 install at:

    C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe

Third-party packages are installed with `python -m pip install --user ...`
(user site: `C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages`),
so **no PYTHONPATH is needed** for any script in this workspace.

## Core stack (already installed)

| package | version | use |
|---|---|---|
| numpy | 2.2.6 | arrays |
| pandas | 3.0.2 | data frames (arrow-backed strings) |
| pyarrow | 26.0.0 | parquet read/write, arrow strings |
| scipy | 1.15.3 | sparse solvers, nnls, ndimage |
| scikit-learn | 1.9.1 | metrics / auxiliary models |
| lightgbm | 4.6.0 | gradient boosting (Task 2, Task 1 residual model) |
| torch | 2.11.0+cpu | optional sequence models |
| matplotlib | 3.10.7 | diagnostics (optional) |
| psutil | 7.2.2 | memory guards (RAM is tight: ~2 GB free) |
| kaggle | 2.2.5 | Kaggle API client (`api.competition_submit`) |
| kagglesdk | 0.1.38 | transport used by the kaggle client |

The workspace also has a legacy wheel-extraction directory `libs/` used earlier
when `pip` was blocked by the DSH sandbox; it is only a fallback and should not
be put on `PYTHONPATH` (removed once the pip install was verified).

## Install from scratch on a clean machine

```powershell
python -m pip install --user -r work/setup/requirements.txt
```

## Verify

```powershell
python work/setup/verify_env.py
```

This checks imports/versions, that the release data and the per-panel caches
exist, that pyarrow can read a release parquet file, and that the Kaggle API
authenticates with `~/.kaggle/access_token`.

## Kaggle submission

Credentials live in `C:\Users\lenovo\.kaggle\access_token` (KGAT_ token,
bearer auth). Competition: `2026-ieee-big-data-traffic-flow-bench`.
Helper: `python work/tools/submit.py --file <file> --message <msg>`.
Max 2 final submissions are selected from daily submissions; quota is 5/day.
