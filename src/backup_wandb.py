from pathlib import Path
import json
import re
import pandas as pd
import wandb


ENTITY = ""
PROJECT = ""
BACKUP_DIR = Path("")


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def export_run(run, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    history = run.history(pandas=True)
    history.to_parquet(output_dir / "history.parquet", index=False)
    metadata = {
        "id": run.id,
        "name": run.name,
        "group": run.group,
        "job_type": run.job_type,
        "state": run.state,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "tags": run.tags,
        "url": run.url,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    with open(output_dir / "config.json", "w") as f:
        json.dump(dict(run.config), f, indent=4)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(dict(run.summary), f, indent=4)


def local_updated_at(run_dir: Path):
    metadata_file = run_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        return metadata.get("updated_at")
    except Exception:
        return None


def sync_project(entity: str, project: str, backup_dir: Path):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    new_runs = 0
    updated_runs = 0
    skipped = 0
    for run in runs:
        group = safe_name(run.group or "ungrouped")
        run_name = safe_name(run.name)
        run_dir = backup_dir / group / run_name
        local_timestamp = local_updated_at(run_dir)
        remote_timestamp = run.updated_at
        if local_timestamp is None:
            print(f"[NEW]      {group}/{run_name}")
            export_run(run, run_dir)
            new_runs += 1
        elif local_timestamp != remote_timestamp:
            print(f"[UPDATED]  {group}/{run_name}")
            export_run(run, run_dir)
            updated_runs += 1
        else:
            skipped += 1
    print("\nSync complete.")
    print(f"New runs      : {new_runs}")
    print(f"Updated runs  : {updated_runs}")
    print(f"Skipped runs  : {skipped}")


if __name__ == "__main__":
    sync_project(ENTITY, PROJECT, BACKUP_DIR)




'''
### Resulting folder structure

```
wandb_backup/
├── ResNet50/
│   ├── lr1e-3_seed1/
│   │   ├── history.parquet
│   │   ├── metadata.json
│   │   ├── config.json
│   │   └── summary.json
│   └── lr1e-3_seed2/
├── EfficientNet/
│   ├── baseline/
│   └── augmented/
└── ungrouped/
```

This implementation is incremental:

* **New run** → downloaded.
* **Existing run with a different `updated_at`** → overwritten with the latest data.
* **Existing run with the same `updated_at`** → skipped.

This makes repeated syncs efficient while ensuring runs that were resumed or continued are kept up to date.
'''