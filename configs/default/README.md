# Default Configs

Each config corresponds to a different experiment mode. All configs share the same base settings (seed, cache_dir, early stopping, etc.) but differ in task-specific parameters.

| Config | Description |
|--------|-------------|
| `basic.yaml` | Entity-level tasks (e.g., node classification/regression) |
| `link.yaml` | Link prediction tasks |
| `random.yaml` | Randomly initialized GNN (no pretrained weights) |
| `tabnn.yaml` | Tabular neural network models (e.g., FT-Transformer) |
| `task.yaml` | Task signature / task embedding experiments |

## Sensitive Fields

The following fields have been moved to the project root `.env` file and are loaded at runtime:

- `MONGODB_URI` — MongoDB connection string (was `db_location`)
- `MONGODB_DB_NAME` — MongoDB database name (was `db_name`)
- `PK` — Public key (was `pk`)
- `PRIVATE_KEY` — Private key (was `private_key`)
