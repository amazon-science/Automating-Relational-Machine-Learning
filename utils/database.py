import json
import hashlib
import sqlite3
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, List, Union, Dict, Any, Literal

try:
    import pymongo
except ImportError:  # pragma: no cover - pymongo is optional when using sqlite backend
    pymongo = None

try:
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _np = None

class ModelType(Enum):
    RDL = "rdl"
    DFS = "dfs"
    NOG = "nograph"

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"

class PreSFType(Enum):
    SRC_DST = "src_dst"
    ZERO_LEARN = "zero_learn"
    NONE = "none"

class MPNNType(Enum):
    SPARSE = "sparse"
    HIGH = "high"
    DENSE = "dense"

class MPNNDesignChoice(Enum):
    # Sparse choices
    SAGE = "sage"
    HGT = "hgt"
    PNA = "pna"
    GRIFFIN = "griffin"
    # High choices
    RELGNN = "relgnn"
    # Dense choices
    RELGT = "relgt"

class PostSFType(Enum):
    NCN = "ncn"
    CN = "cn"
    NONE = "none"
    SHALLOW = "shallow"

class LossFunction(Enum):
    CE = "ce"
    MSE = "mse"
    MAE = "mae"
    BPR = "bpr"

class NormType(Enum):
    LAYERNORM = "layernorm"
    BATCHNORM = "batchnorm"
    NONE = "none"

class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"

class TemporalType(Enum):
    UNIFORM = "uniform"
    LAST = "last"

class LoaderType(Enum):
    NODE = "node"
    EDGE = "edge"

class SchedulerType(Enum):
    EXPONENTIAL = "expontential"
    NONE = "none"

class TabModel(Enum):
    TABPFN = "tabpfn"
    RESNET = "resnet"
    FT_TRANSFORMER = "ft_transformer"
    LGBM = "lightgbm"



@dataclass 
class AutoTabConfig:
    max_train_samples: Optional[int] = None
    # Strategy for selecting training samples.
    training_sample_selection_strategy: Literal['graph', 'random', 'stratified'] = 'random'
    # Whether to use foreign keys as features.
    negative_sampling_ratio: int = 5
    # num_trials
    num_trials: int = 30
    
    

@dataclass
class RDLConfig:
    pre_sf: PreSFType
    mpnn_type: MPNNType
    mpnn_design_choice: MPNNDesignChoice
    post_sf: PostSFType
    loss_fn: LossFunction
    hidden_channels: int
    num_heads: int
    dropout: float
    norm: NormType
    aggregation: AggregationType
    jk: bool
    skip_connections: bool

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create RDLConfig instance from dictionary with string values"""
        return cls(
            pre_sf=PreSFType(config_dict["gnn_config"]["pre_sf"]),
            mpnn_type=MPNNType(config_dict["gnn_config"]["mpnn_type"]),
            mpnn_design_choice=MPNNDesignChoice(config_dict["gnn_config"]["mpnn_design_choice"]),
            post_sf=PostSFType(config_dict["gnn_config"]["post_sf"]),
            loss_fn=LossFunction(config_dict["gnn_config"]["loss_fn"]),
            hidden_channels=config_dict["gnn_config"]["hidden_channels"],
            num_heads=config_dict["gnn_config"]["num_heads"],
            dropout=config_dict["gnn_config"]["dropout"],
            norm=NormType(config_dict["gnn_config"]["norm"]),
            aggregation=AggregationType(config_dict["gnn_config"]["aggregation"]),
            jk=config_dict["gnn_config"]["jk"],
            skip_connections=config_dict["gnn_config"]["skip_connections"]
        )

@dataclass
class SamplerConfig:
    temporal: TemporalType
    num_neighbors: int
    num_layers: int
    loader_type: LoaderType

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create SamplerConfig instance from dictionary with string values"""
        return cls(
            temporal=TemporalType(config_dict["sampler_config"]["temporal"]),
            num_neighbors=config_dict["sampler_config"]["num_neighbors"],
            num_layers=config_dict["sampler_config"]["num_layers"],
            loader_type=LoaderType(config_dict["sampler_config"]["loader_type"])
        )

@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    scheduler: SchedulerType
    gamma: float

    @classmethod

    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create OptimizerConfig instance from dictionary with string values"""
        return cls(
            lr=config_dict["optimizer_config"]["lr"],
            weight_decay=config_dict["optimizer_config"]["weight_decay"],
            scheduler=SchedulerType(config_dict["optimizer_config"]["scheduler"]),
            gamma=config_dict["optimizer_config"]["gamma"])

@dataclass 
class TabNNConfig:
    hidden_size : Optional[int] = 128
    dropout : Optional[float] = 0.7
    num_layers : Optional[int] = 3
    attn_dropout : Optional[float] = 0.0
    num_heads : Optional[int] = 8
    normalization : Optional[str] = "layer_norm"
    optimizer_config: Optional[OptimizerConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(
            hidden_size=config_dict["hidden_size"],
            dropout=config_dict["dropout"],
            num_layers=config_dict["num_layers"],
            attn_dropout=config_dict["attn_dropout"],
            num_heads=config_dict["num_heads"],
            normalization=config_dict["normalization"]
        )

@dataclass
class RDLModelConfig:
    encoder_num_layers: int
    channels: int
    torch_frame_model_cls: str
    rdl_config: RDLConfig
    sampler_config: SamplerConfig
    optimizer_config: OptimizerConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create ModelConfig instance from dictionary with string values"""
        return cls(
            encoder_num_layers=config_dict["encoder_num_layers"],
            channels=config_dict["channels"],
            torch_frame_model_cls=config_dict["torch_frame_model_cls"],
            rdl_config=RDLConfig.from_dict(config_dict["rdl_config"]),
            sampler_config=SamplerConfig.from_dict(config_dict["sampler_config"]),
            optimizer_config=OptimizerConfig.from_dict(config_dict["optimizer_config"])
        )

@dataclass 
class DFSModelConfig:
    tab_model: TabModel
    tab_nn_config: Union[TabNNConfig, AutoTabConfig]
    dfs_layer: int

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(
            tab_model=TabModel(config_dict["tab_model"]),
            tab_nn_config=TabNNConfig.from_dict(config_dict["tab_nn_config"]),
            dfs_layer=config_dict["dfs_layer"]
        )

@dataclass
class NOGModelConfig:
    tab_model: TabModel
    tab_nn_config: Union[TabNNConfig, AutoTabConfig]
    nog_layer: int
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(
            tab_model=TabModel(config_dict["tab_model"]),
            tab_nn_config=TabNNConfig.from_dict(config_dict["tab_nn_config"]),
            nog_layer=config_dict["nog_layer"]
        )

@dataclass 
class ClassificationResult:
    acc: float
    f1: float 
    roc_auc: float 

@dataclass
class RegressionResult:
    rmse: float
    mae: float 
    r2: float 

@dataclass
class RecommendationResult:
    ndcg: float
    recall: float
    precision: float
    map: float
    mrr: float

class Status(Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    MEMORY_OUT = "memory_out"
    RUNNING = "running"

class ResultType(Enum):
    TRAIN_LOSS = "train_loss"
    VAL_METRIC = "val_metric"
    TEST_METRIC = "test_metric"


@dataclass
class SearchRow:
    task_name: str
    db_name: str 
    model_type: ModelType
    params: Union[RDLModelConfig, DFSModelConfig, NOGModelConfig]  # RDLParams for RDL, Dict for others
    results: Union[ClassificationResult, RegressionResult, RecommendationResult]
    status: Status

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create SearchRow instance from dictionary with string values"""
        return cls(
            task_name=config_dict["task_name"],
            db_name=config_dict["db_name"],
            model_type=ModelType(config_dict["model_type"]),
            params=RDLModelConfig.from_dict(config_dict["params"]),
            results=config_dict["results"],
            status=Status(config_dict["status"])
        )

class NullSwapDB:
    """A no-op replacement for SwapDB that requires no MongoDB connection."""

    def __init__(self):
        self.test = True

    def add_a_row(self, *args, **kwargs):
        pass

    def get_all_rows(self):
        return []

    def get_rows_with_the_same_search_params(self, *args, **kwargs):
        return []

    def find_same_arch_gpu_out_rows(self, *args, **kwargs):
        return []

    def count_trials_matching_search(self, *args, **kwargs):
        return 0

    def count_trials_matching_search_with_status(self, *args, **kwargs):
        return 0

    def save_loss_metrics(self, *args, **kwargs):
        pass

    def search_loss_metrics(self, *args, **kwargs):
        return False

    def save_trial_artifacts(self, *args, **kwargs):
        pass

    def find_trial_artifacts_by_config(self, *args, **kwargs):
        return []

    def find_loss_metrics_by_config(self, *args, **kwargs):
        return []


# ---------------------------------------------------------------------------
# JSON helpers shared by both backends
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except Exception:
            return obj.hex()
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _config_hash(config: Dict[str, Any]) -> str:
    try:
        serialized = json.dumps(config, sort_keys=True, default=_json_default)
    except TypeError:
        serialized = json.dumps(_json_default(config), sort_keys=True, default=_json_default)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _doc_matches_flat(doc: Dict[str, Any], flat_query: Dict[str, Any]) -> bool:
    """Check whether *doc* satisfies every key=value in *flat_query*."""
    flat_doc = _flatten_dict(doc)
    for key, value in flat_query.items():
        if flat_doc.get(key) != value:
            return False
    return True


# ---------------------------------------------------------------------------
# SQLite-backed SwapDB – drop-in replacement that needs no MongoDB server
# ---------------------------------------------------------------------------
class SQLiteSwapDB:
    """A local SQLite replacement for SwapDB.

    Documents are stored as JSON blobs in a single ``documents`` table.
    The config hash is persisted as a column for fast exact-match lookups.
    """

    def __init__(self, path_to_db: str, collection_name: str, db_name: str = 'swap'):
        # *path_to_db* is reused as a directory; the actual file lives inside.
        db_dir = os.path.abspath(path_to_db)
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, f"{db_name}_{collection_name}.sqlite")
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  config_hash TEXT,"
            "  doc TEXT NOT NULL"
            ")"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_config_hash ON documents(config_hash)"
        )
        self.conn.commit()
        self.test = False

    # -- helpers ---------------------------------------------------------------
    def _insert(self, doc: Dict[str, Any]):
        config = doc.get('config', {})
        ch = _config_hash(config) if config else ''
        blob = json.dumps(doc, default=_json_default)
        self.conn.execute(
            "INSERT INTO documents (config_hash, doc) VALUES (?, ?)",
            (ch, blob),
        )
        self.conn.commit()

    def _all_docs(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT doc FROM documents").fetchall()
        return [json.loads(r[0]) for r in rows]

    def _find_by_config_hash(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        ch = _config_hash(config)
        rows = self.conn.execute(
            "SELECT doc FROM documents WHERE config_hash = ?", (ch,)
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # -- public API (mirrors SwapDB) -------------------------------------------
    def add_a_row(self, config: Dict[str, Any], result: Dict[str, Any], status, epoch: int = 0):
        status_val = status.value if isinstance(status, Status) else str(status)
        self._insert({
            "config": config,
            "results": result,
            "status": status_val,
            "epoch": epoch,
            "result_type": "test_metric",
        })

    def get_all_rows(self):
        return self._all_docs()

    def get_rows_with_the_same_search_params(self, searched_config: Dict[str, Any]):
        if self.test:
            return []
        finished = {Status.COMPLETED.value, Status.MEMORY_OUT.value}
        flat_query = {f"config.{k}": v for k, v in _flatten_dict(searched_config).items()}
        results = []
        for doc in self._all_docs():
            if doc.get("status") not in finished:
                continue
            if _doc_matches_flat(doc, flat_query):
                results.append(doc)
        return results

    def find_same_arch_gpu_out_rows(self, searched_config: Dict[str, Any]):
        if self.test:
            return []
        mp = searched_config.get('model_params', {})
        if mp.get('model_type') != 'rdl':
            return []
        results = []
        for doc in self._all_docs():
            if doc.get('status') != Status.MEMORY_OUT.value:
                continue
            dc = doc.get('config', {})
            dmp = dc.get('model_params', {})
            dgnn = dmp.get('gnn_config', {})
            sgnn = mp.get('gnn_config', {})
            if (dc.get('db_name') != searched_config.get('db_name') or
                dc.get('task_name') != searched_config.get('task_name')):
                continue
            arch_match = (
                dgnn.get('pre_sf') == sgnn.get('pre_sf') and
                dgnn.get('mpnn_type') == sgnn.get('mpnn_type') and
                dgnn.get('post_sf') == sgnn.get('post_sf') and
                dgnn.get('jk') == sgnn.get('jk') and
                dgnn.get('skip_connection') == sgnn.get('skip_connection')
            )
            if not arch_match:
                continue
            fidelity_ok = (
                dgnn.get('hidden_channels', 0) <= sgnn.get('hidden_channels', 0) and
                dmp.get('sampler_config', {}).get('num_layers', 0) <= mp.get('sampler_config', {}).get('num_layers', 0) and
                dmp.get('sampler_config', {}).get('num_neighbors', 0) <= mp.get('sampler_config', {}).get('num_neighbors', 0) and
                dmp.get('batch_size', 0) <= mp.get('batch_size', 0)
            )
            if fidelity_ok:
                results.append(doc)
        return results

    def count_trials_matching_search(self, search_config: Dict[str, Any]) -> int:
        if self.test:
            return 0
        flat_query = {f"config.{k}": v for k, v in _flatten_dict(search_config).items()}
        return sum(1 for doc in self._all_docs() if _doc_matches_flat(doc, flat_query))

    def count_trials_matching_search_with_status(self, search_config: Dict[str, Any], status_filter=None) -> int:
        if self.test:
            return 0
        flat_query = {f"config.{k}": v for k, v in _flatten_dict(search_config).items()}
        status_vals = {s.value for s in status_filter} if status_filter else None
        count = 0
        for doc in self._all_docs():
            if status_vals and doc.get('status') not in status_vals:
                continue
            if _doc_matches_flat(doc, flat_query):
                count += 1
        return count

    def save_loss_metrics(self, metrics: Dict[str, Any]):
        metrics = dict(metrics)
        metrics.setdefault('artifact_type', 'loss_metrics')
        self._insert(metrics)

    def search_loss_metrics(self, task_name: str, type_f: str):
        for doc in self._all_docs():
            if doc.get('task_name') == task_name and doc.get('type') == type_f:
                return True
        return False

    def save_trial_artifacts(self, artifact_doc: Dict[str, Any]):
        artifact_doc = dict(artifact_doc)
        artifact_doc.setdefault('artifact_type', 'trial_artifacts')
        self._insert(artifact_doc)

    def find_trial_artifacts_by_config(self, model_config: Dict[str, Any]) -> list:
        if self.test:
            return []
        mp = model_config.get('model_params', {})
        dbs = model_config.get('dbs', [])
        db_name = task_name = None
        if isinstance(dbs, list) and dbs:
            db_name = dbs[0].get('db_name')
            tn = dbs[0].get('task_name')
            if isinstance(tn, list) and tn:
                task_name = tn[0]
            elif isinstance(tn, str):
                task_name = tn

        rdl_config = mp.get('torch_frame_model_cls') == 'resnet'
        if rdl_config:
            check_keys = [
                ('config.model_params.encoder_num_layers', mp.get('encoder_num_layers')),
                ('config.model_params.torch_frame_model_cls', mp.get('torch_frame_model_cls')),
                ('config.model_params.gnn_config.pre_sf', mp.get('gnn_config', {}).get('pre_sf')),
                ('config.model_params.gnn_config.mpnn_type', mp.get('gnn_config', {}).get('mpnn_type')),
                ('config.model_params.gnn_config.hidden_channels', mp.get('gnn_config', {}).get('hidden_channels')),
                ('config.model_params.gnn_config.num_heads', mp.get('gnn_config', {}).get('num_heads')),
                ('config.model_params.gnn_config.dropout', mp.get('gnn_config', {}).get('dropout')),
                ('config.model_params.gnn_config.norm', mp.get('gnn_config', {}).get('norm')),
                ('config.model_params.gnn_config.aggregation', mp.get('gnn_config', {}).get('aggregation')),
                ('config.model_params.sampler_config.temporal', mp.get('sampler_config', {}).get('temporal')),
                ('config.model_params.sampler_config.num_neighbors', mp.get('sampler_config', {}).get('num_neighbors')),
                ('config.model_params.sampler_config.num_layers', mp.get('sampler_config', {}).get('num_layers')),
                ('config.model_params.optimizer_config.lr', mp.get('optimizer_config', {}).get('lr')),
                ('config.model_params.optimizer_config.weight_decay', mp.get('optimizer_config', {}).get('weight_decay')),
                ('config.model_params.optimizer_config.gamma', mp.get('optimizer_config', {}).get('gamma')),
            ]
        else:
            check_keys = [
                ('config.model_params.dfs_layer', mp.get('dfs_layer')),
                ('config.model_params.torch_frame_model_cls', mp.get('torch_frame_model_cls')),
                ('config.model_params.batch_size', mp.get('batch_size')),
                ('config.model_params.tab_nn_config.hidden_size', mp.get('tab_nn_config', {}).get('hidden_size')),
                ('config.model_params.tab_nn_config.dropout', mp.get('tab_nn_config', {}).get('dropout')),
                ('config.model_params.tab_nn_config.num_layers', mp.get('tab_nn_config', {}).get('num_layers')),
                ('config.model_params.tab_nn_config.attn_dropout', mp.get('tab_nn_config', {}).get('attn_dropout')),
                ('config.model_params.tab_nn_config.num_heads', mp.get('tab_nn_config', {}).get('num_heads')),
                ('config.model_params.tab_nn_config.normalization', mp.get('tab_nn_config', {}).get('normalization')),
                ('config.model_params.tab_nn_config.optimizer_config.lr', mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('lr')),
                ('config.model_params.tab_nn_config.optimizer_config.weight_decay', mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('weight_decay')),
                ('config.model_params.tab_nn_config.optimizer_config.gamma', mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('gamma')),
            ]

        flat_query = dict(check_keys)
        results = []
        for doc in self._all_docs():
            flat_doc = _flatten_dict(doc)
            match = True
            for key, val in flat_query.items():
                if flat_doc.get(key) != val:
                    match = False
                    break
            if not match:
                continue
            if db_name and task_name:
                cfg_dbs = doc.get('config', {}).get('dbs', [])
                found = False
                for entry in (cfg_dbs if isinstance(cfg_dbs, list) else []):
                    if entry.get('db_name') == db_name:
                        tn = entry.get('task_name', [])
                        if task_name in (tn if isinstance(tn, list) else [tn]):
                            found = True
                            break
                if not found:
                    continue
            results.append(doc)
        return results

    def find_loss_metrics_by_config(self, model_config: Dict[str, Any], tag: str = None) -> list:
        if self.test:
            return []
        mp = model_config.get('model_params', {})
        rdl_config = mp.get('torch_frame_model_cls') == 'resnet'
        if rdl_config:
            check_keys = [
                ('config.model_params.encoder_num_layers', mp.get('encoder_num_layers')),
                ('config.model_params.torch_frame_model_cls', mp.get('torch_frame_model_cls')),
                ('config.model_params.gnn_config.pre_sf', mp.get('gnn_config', {}).get('pre_sf')),
                ('config.model_params.gnn_config.mpnn_type', mp.get('gnn_config', {}).get('mpnn_type')),
                ('config.model_params.gnn_config.hidden_channels', mp.get('gnn_config', {}).get('hidden_channels')),
                ('config.model_params.gnn_config.num_heads', mp.get('gnn_config', {}).get('num_heads')),
                ('config.model_params.gnn_config.dropout', mp.get('gnn_config', {}).get('dropout')),
                ('config.model_params.gnn_config.norm', mp.get('gnn_config', {}).get('norm')),
                ('config.model_params.gnn_config.aggregation', mp.get('gnn_config', {}).get('aggregation')),
                ('config.model_params.sampler_config.temporal', mp.get('sampler_config', {}).get('temporal')),
                ('config.model_params.sampler_config.num_neighbors', mp.get('sampler_config', {}).get('num_neighbors')),
                ('config.model_params.sampler_config.num_layers', mp.get('sampler_config', {}).get('num_layers')),
                ('config.model_params.sampler_config.loader_type', mp.get('sampler_config', {}).get('loader_type')),
                ('config.model_params.optimizer_config.lr', mp.get('optimizer_config', {}).get('lr')),
                ('config.model_params.optimizer_config.weight_decay', mp.get('optimizer_config', {}).get('weight_decay')),
                ('config.model_params.optimizer_config.scheduler', mp.get('optimizer_config', {}).get('scheduler')),
                ('config.model_params.optimizer_config.gamma', mp.get('optimizer_config', {}).get('gamma')),
            ]
        else:
            check_keys = [
                ('config.model_params.dfs_layer', mp.get('dfs_layer')),
                ('config.model_params.torch_frame_model_cls', mp.get('torch_frame_model_cls')),
                ('config.model_params.batch_size', mp.get('batch_size')),
                ('config.model_params.tab_nn_config.hidden_size', mp.get('tab_nn_config', {}).get('hidden_size')),
                ('config.model_params.tab_nn_config.dropout', mp.get('tab_nn_config', {}).get('dropout')),
                ('config.model_params.tab_nn_config.num_layers', mp.get('tab_nn_config', {}).get('num_layers')),
                ('config.model_params.tab_nn_config.attn_dropout', mp.get('tab_nn_config', {}).get('attn_dropout')),
                ('config.model_params.tab_nn_config.num_heads', mp.get('tab_nn_config', {}).get('num_heads')),
                ('config.model_params.tab_nn_config.normalization', mp.get('tab_nn_config', {}).get('normalization')),
                ('config.model_params.tab_nn_config.optimizer_config.lr', mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('lr')),
                ('config.model_params.tab_nn_config.optimizer_config.weight_decay', mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('weight_decay')),
                ('config.model_params.tab_nn_config.optimizer_config.scheduler', mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('scheduler')),
                ('config.model_params.tab_nn_config.optimizer_config.gamma', mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('gamma')),
            ]
        flat_query = dict(check_keys)
        results = []
        for doc in self._all_docs():
            flat_doc = _flatten_dict(doc)
            match = True
            for key, val in flat_query.items():
                if flat_doc.get(key) != val:
                    match = False
                    break
            if match:
                results.append(doc)
        return results


def create_swap_db(backend: str = 'sqlite', path_to_db: str = './swap_db',
                   collection_name: str = 'default', db_name: str = 'swap') -> Any:
    """Factory to create a SwapDB-compatible backend.

    Args:
        backend: ``'sqlite'`` (default, local), ``'mongodb'``, or ``'null'``.
        path_to_db: For sqlite – directory where .sqlite files are stored.
                    For mongodb – the MongoDB connection URI.
        collection_name: Logical collection / table name.
        db_name: Database name (used as prefix in sqlite filenames).

    Returns:
        An object with the SwapDB interface.
    """
    backend = (backend or 'sqlite').lower().strip()
    if backend == 'null' or backend == 'none':
        return NullSwapDB()
    if backend == 'sqlite':
        return SQLiteSwapDB(path_to_db=path_to_db, collection_name=collection_name, db_name=db_name)
    if backend in ('mongodb', 'mongo'):
        if pymongo is None:
            raise ImportError("pymongo is required for the mongodb backend. Install it or use backend='sqlite'.")
        return SwapDB(path_to_db=path_to_db, collection_name=collection_name, db_name=db_name)
    raise ValueError(f"Unknown db backend: {backend!r}. Choose 'sqlite', 'mongodb', or 'null'.")


class SwapDB:
    def __init__(self, path_to_db: str, collection_name: str, db_name: str = 'swap'):
        self.client = pymongo.MongoClient(path_to_db)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.test = False

    def add_a_row(self, config: Dict[str, Any], result: Dict[str, Any], status: Status, epoch: int = 0):
        """Add a search row to the database"""
        full_dict = {
            "config": config,
            "results": result,
            "status": status.value if isinstance(status, Status) else status,
            "epoch": epoch,
            "result_type": 'test_metric'
        }
        
        return self.collection.insert_one(full_dict)

    def get_all_rows(self):
        """Retrieve all search rows from the database"""
        return list(self.collection.find())

    def get_rows_with_the_same_search_params(self, searched_config: Dict[str, Any]):
        """
        Finds all rows with the same parameters that have already been run
        (i.e., are not in a 'running' state), matching recursively.
        """
        if self.test:
            return []
        # if searched_config['model_params']['model_type'] != 'rdl':
        #     return []
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Define the statuses that indicate a run is finished.
        # We use .value to get the string representation (e.g., "completed").
        finished_statuses = [
            Status.COMPLETED.value,
            Status.MEMORY_OUT.value
        ]

        # Build the query with dot notation for nested fields
        query_params = {}
        for key, value in flatten_dict(searched_config).items():
            query_params[f"config.{key}"] = value

        # The query now checks if the 'status' field is in our list of finished statuses.
        query = {
            **query_params,
            "status": {"$in": finished_statuses}
        }

        return list(self.collection.find(query))
    
    def remove_data_of_a_task(self, database_name: str, task_name: str):
        """
        Remove all data of a task from the database
        """
        self.collection.delete_many({"config.db_name": database_name, "config.task_name": task_name})
    
    def find_same_arch_gpu_out_rows(self, searched_config: Dict[str, Any]):
        """
        Retrieve rows with the same model architecture that previously failed due to GPU memory out.

        This function is designed to proactively identify configurations that are likely to fail
        due to insufficient GPU memory. It queries for past runs that:
        1. Match the core architectural parameters of the `searched_config`.
        2. Failed with a 'memory_out' status.
        3. Used fidelity-related parameters (like hidden size, number of layers, batch size)
           that are less than or equal to the current `searched_config`.

        If such a row is found, it suggests the current configuration is also at high risk
        of failing.
        """
        # The query is constructed assuming the configuration is for an RDL model,
        # as it references specific RDL and sampler configurations.
        # Paths in the query use dot notation to access nested fields within the 'config'
        # object stored in the MongoDB collection.
        if self.test:
            return []
        if searched_config['model_params']['model_type'] != 'rdl':
            return []
        query = {
            "$and": [
                # --- Match the same model architecture ---
                {"config.db_name": {"$eq": searched_config["db_name"]}},
                {"config.task_name": {"$eq": searched_config["task_name"]}},
                {"config.model_params.gnn_config.pre_sf": {"$eq": searched_config['model_params']["gnn_config"]["pre_sf"]}},
                {"config.model_params.gnn_config.mpnn_type": {"$eq": searched_config['model_params']["gnn_config"]["mpnn_type"]}},
                {"config.model_params.gnn_config.post_sf": {"$eq": searched_config["model_params"]["gnn_config"]["post_sf"]}},
                {"config.model_params.gnn_config.jk": {"$eq": searched_config["model_params"]["gnn_config"]["jk"]}},
                {"config.model_params.gnn_config.skip_connection": {"$eq": searched_config["model_params"]["gnn_config"]["skip_connection"]}},

                # --- Check for less than or equal fidelity-related parameters ---
                {"config.model_params.gnn_config.hidden_channels": {"$lte": searched_config["model_params"]["gnn_config"]["hidden_channels"]}},
                {"config.model_params.sampler_config.num_layers": {"$lte": searched_config["model_params"]["sampler_config"]["num_layers"]}},
                {"config.model_params.sampler_config.num_neighbors": {"$lte": searched_config["model_params"]["sampler_config"]["num_neighbors"]}},
                {"config.model_params.encoder_num_layers": {"$lte": searched_config["model_params"]["encoder_num_layers"]}},
                
                # Assuming 'batch_size' and 'eval_batch_size' are present in the config.
                # If not, these conditions might need to be handled differently (e.g., using $exists).
                {"config.model_params.batch_size": {"$lte": searched_config["model_params"]["batch_size"]}},
                {"config.model_params.eval_batch_size": {"$lte": searched_config["model_params"]["eval_batch_size"]}},

                # --- Check for the specific 'memory_out' status ---
                {"status": {"$eq": Status.MEMORY_OUT.value}}
            ]
        }
        return list(self.collection.find(query))

    def get_num_of_finished_rows(self, database_name: str, task_name: str, pre_sf: str = "", mpnn_type: str = "", post_sf: str = ""):
        """
        Get the number of finished rows (completed status) matching the specified parameters.
        
        Args:
            pre_sf: Pre-sampling function type
            mpnn_type: MPNN type
            post_sf: Post-sampling function type
            
        Returns:
            int: Number of finished rows matching the criteria
        """            
        # Query for rows with completed status and matching parameters
        query = {
    "config.dbs": {
        "$elemMatch": {
            "db_name": database_name,           
            "task_name": {"$in": [task_name]}   
        }
    },
    "status": {"$in": ["completed", "COMPLETED"]},
    "result_type": {"$in": ["test_metric"]}          # 
}
        if pre_sf != "":
            query["config.model_params.gnn_config.pre_sf"] = pre_sf
        if mpnn_type != "":
            query["config.model_params.gnn_config.mpnn_type"] = mpnn_type
        if post_sf != "":
            query["config.model_params.gnn_config.post_sf"] = post_sf
        
        # Count documents matching the query
        count = self.collection.count_documents(query)
        
        return count
    
    def get_num_of_finished_rows_with_tabnn(self, database_name: str, task_name: str, dfs_layer: str = ""):
        """
        Get the number of finished rows (completed status) matching the specified parameters.
        
        Args:
            pre_sf: Pre-sampling function type
            mpnn_type: MPNN type
            post_sf: Post-sampling function type
            
        Returns:
            int: Number of finished rows matching the criteria
        """            
        # Query for rows with completed status and matching parameters
        query = {
            "config.dbs": {
                "$elemMatch": {
                    "db_name": database_name,           
                    "task_name": {"$in": [task_name]}   
                }
            },
            "status": {"$in": ["completed", "COMPLETED"]},
            "result_type": {"$in": ["test_metric"]}          # 
        }
        query["config.model_params.dfs_layer"] = dfs_layer
        
        # Count documents matching the query
        count = self.collection.count_documents(query)
        
        return count
    
    def get_num_of_finished_rows_with_tree(self, database_name: str, task_name: str, dfs_layer: str = "", model_name: str = "", text_to_pca: Union[bool, str] = ""):
        """
        Get the number of finished rows (completed status) matching the specified parameters.
        
        Args:
            dfs_layer: DFS layer
            model_name: Model name
        """
        query = {
            "config.dbs": {
                "$elemMatch": {
                    "db_name": database_name,           
                    "task_name": {"$in": [task_name]}   
                }
            },
            "status": {"$in": ["completed", "COMPLETED"]},
            "result_type": {"$in": ["test_metric"]}          # 
        }
        if dfs_layer != "":
            query["config.model_params.dfs_layer"] = dfs_layer
        if model_name != "":
            query["config.model_params.torch_frame_model_cls"] = model_name
        if text_to_pca != "":
            query["config.model_params.text_to_pca"] = text_to_pca
        count = self.collection.count_documents(query)
        return count
    
    def get_finished_rows_with_rdl(self, task_name: str):
        query = {
            "config.dbs": {
                "$elemMatch": {          
                    "task_name": {"$in": [task_name]}   
                }
            },
            "config.model_params.model_type": {"$in": ["rdl"]},
            "status": {"$in": ["completed", "COMPLETED"]},
            "result_type": {"$in": ["test_metric"]}          # 
        }
        return list(self.collection.find(query))
    
    def remove_rdl_trials(self, task_name: str, mpnn_type: str):
        query = {
            "config.dbs": {
                "$elemMatch": {          
                    "task_name": {"$in": [task_name]}   
                }
            },
            "config.model_params.gnn_config.mpnn_type": {"$in": [mpnn_type]},
            "config.model_params.model_type": {"$in": ["rdl"]},
        }
        self.collection.delete_many(query)
    
    def get_finished_rows_with_tree(self, task_name: str, dfs_layer: str = "", model_name: str = "", text_to_pca: Union[bool, str] = ""):
        query = {
            "config.dbs": {
                "$elemMatch": {          
                    "task_name": {"$in": [task_name]}   
                }
            },
            "status": {"$in": ["completed", "COMPLETED"]},
            "result_type": {"$in": ["test_metric"]}          # 
        }
        if dfs_layer != "":
            query["config.model_params.dfs_layer"] = dfs_layer
        if model_name != "":
            query["config.model_params.torch_frame_model_cls"] = model_name
        if text_to_pca != "":
            query["config.model_params.text_to_pca"] = text_to_pca
        return list(self.collection.find(query))
    
    def get_finished_rows_with_tabnn(self, task_name: str, dfs_layer: str = ""):
        query = {
            "config.dbs": {
                "$elemMatch": {          
                    "task_name": {"$in": [task_name]}   
                }
            },
            "status": {"$in": ["completed", "COMPLETED"]},
            "result_type": {"$in": ["test_metric"]}          # 
        }
        query["config.model_params.dfs_layer"] = dfs_layer
        return list(self.collection.find(query))


    def count_trials_matching_search(self, search_config: Dict[str, Any]) -> int:
        """
        Find the number of trials matching the search input configuration.
        
        Args:
            search_config: Dictionary containing the search parameters to match against
            
        Returns:
            int: Number of trials that match the search configuration
        """
        if self.test:
            return 0
            
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.'):
            """Flatten nested dictionary for query construction"""
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Build the query with dot notation for nested fields
        query_params = {}
        for key, value in flatten_dict(search_config).items():
            query_params[f"config.{key}"] = value

        # Count documents matching the search configuration
        count = self.collection.count_documents(query_params)
        
        return count

    def count_trials_matching_search_with_status(self, search_config: Dict[str, Any], status_filter: Optional[List[Status]] = None) -> int:
        """
        Find the number of trials matching the search input configuration with optional status filtering.
        
        Args:
            search_config: Dictionary containing the search parameters to match against
            status_filter: Optional list of Status enums to filter by. If None, counts all matching trials.
            
        Returns:
            int: Number of trials that match the search configuration and status filter
        """
        if self.test:
            return 0
            
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.'):
            """Flatten nested dictionary for query construction"""
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        # Build the query with dot notation for nested fields
        query_params = {}
        for key, value in flatten_dict(search_config).items():
            query_params[f"config.{key}"] = value

        # Add status filter if provided
        if status_filter is not None:
            status_values = [status.value for status in status_filter]
            query_params["status"] = {"$in": status_values}

        # Count documents matching the search configuration and status filter
        count = self.collection.count_documents(query_params)
        
        return count
    
    def remove_rows_with_buggy_params(self):
        ## when post_sf is contextgnn or rhs is buggy
        query = {
            "$and": [
                {"config.model_params.gnn_config.post_sf": {"$in": ["contextgnn", "rhs"]}}
            ]
        }
        ## count how many documents
        count = self.collection.count_documents(query)
        print(f"Removing {count} rows with buggy params")
        self.collection.delete_many(query)
        return count
    
    def remove_all_rows_with_non_completed_status(self):
        query = {
            "status": {"$ne": Status.COMPLETED.value}
        }
        count = self.collection.count_documents(query)
        print(f"Removing {count} rows with non-completed status")
        self.collection.delete_many(query)
        return count
        
    
    def save_loss_metrics(self, metrics: Dict[str, Any]):
        metrics = dict(metrics)
        config = metrics.get('config', {})
        metrics.setdefault('artifact_type', 'loss_metrics')
        # metrics['config_hash'] = self._config_hash(config)
        ## upload to mongodb
        self.collection.insert_one(metrics)
    
    def search_loss_metrics(self, task_name: str, type_f: str):
        query = {
            'task_name': task_name,
            'type': type_f
        }
        return len(list(self.collection.find(query))) > 0

    # --------- New helpers for artifacts and metric retrieval by config ---------
    def _flatten(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def save_trial_artifacts(self, artifact_doc: Dict[str, Any]):
        """
        Insert a document describing trial artifacts:
        - artifact_type: 'trial_artifacts'
        - config: full model_config used to run
        - results: experiment performance (val/test metrics)
        - model_weight_name: path to saved model
        - optional: any other notes
        """
        artifact_doc = dict(artifact_doc)
        artifact_doc.setdefault('artifact_type', 'trial_artifacts')
        self.collection.insert_one(artifact_doc)

    def find_trial_artifacts_by_config(self, model_config: Dict[str, Any]) -> list:
        """
        Find saved trial artifacts for a given config. Be robust to non-essential
        differences by matching on:
          - artifact_type == 'trial_artifacts'
          - config.model_params equality (exact dict)
          - config.dbs contains matching db_name and task_name
        """
        if self.test:
            return []
        mp = model_config.get('model_params', {})
        dbs = model_config.get('dbs', [])
        rdl_config = mp.get('torch_frame_model_cls', {}) == 'resnet'
        db_name = None
        task_name = None
        if isinstance(dbs, list) and dbs:
            db_name = dbs[0].get('db_name')
            tn = dbs[0].get('task_name')
            if isinstance(tn, list) and tn:
                task_name = tn[0]
            elif isinstance(tn, str):
                task_name = tn
        # query = {
        #     # 'artifact_type': 'trial_artifacts',
        #     'config.model_params': mp,
        # }
        if rdl_config:
            query = {
                '$and': [
                    {'config.model_params.encoder_num_layers': {'$eq': mp.get('encoder_num_layers')}},
                    # {'config.model_params.channels': {'$eq': mp.get('channels')}},
                    {'config.model_params.torch_frame_model_cls': {'$eq': mp.get('torch_frame_model_cls')}},
                    {'config.model_params.gnn_config.pre_sf': {'$eq': mp.get('gnn_config', {}).get('pre_sf')}},
                    {'config.model_params.gnn_config.mpnn_type': {'$eq': mp.get('gnn_config', {}).get('mpnn_type')}},
                    # # {'config.model_params.gnn_config.mpnn_design_choice': {'$eq': mp.get('gnn_config', {}).get('mpnn_design_choice')}},
                    # {'config.model_params.gnn_config.post_sf': {'$eq': mp.get('gnn_config', {}).get('post_sf')}},
                    {'config.model_params.gnn_config.hidden_channels': {'$eq': mp.get('gnn_config', {}).get('hidden_channels')}},
                    {'config.model_params.gnn_config.num_heads': {'$eq': mp.get('gnn_config', {}).get('num_heads')}},
                    {'config.model_params.gnn_config.dropout': {'$eq': mp.get('gnn_config', {}).get('dropout')}},
                    {'config.model_params.gnn_config.norm': {'$eq': mp.get('gnn_config', {}).get('norm')}},
                    {'config.model_params.gnn_config.aggregation': {'$eq': mp.get('gnn_config', {}).get('aggregation')}},
                    # {'config.model_params.gnn_config.jk': {'$eq': mp.get('gnn_config', {}).get('jk')}},
                    {'config.model_params.sampler_config.temporal': {'$eq': mp.get('sampler_config', {}).get('temporal')}},
                    {'config.model_params.sampler_config.num_neighbors': {'$eq': mp.get('sampler_config', {}).get('num_neighbors')}},
                    {'config.model_params.sampler_config.num_layers': {'$eq': mp.get('sampler_config', {}).get('num_layers')}},
                    {'config.model_params.optimizer_config.lr': {'$eq': mp.get('optimizer_config', {}).get('lr')}},
                    {'config.model_params.optimizer_config.weight_decay': {'$eq': mp.get('optimizer_config', {}).get('weight_decay')}},
                    {'config.model_params.optimizer_config.gamma': {'$eq': mp.get('optimizer_config', {}).get('gamma')}},
                ]
            }
        else:
            query = {
                '$and': [
                    {'config.model_params.dfs_layer': {'$eq': mp.get('dfs_layer')}},
                    {'config.model_params.torch_frame_model_cls': {'$eq': mp.get('torch_frame_model_cls')}},
                    {'config.model_params.batch_size': {'$eq': mp.get('batch_size')}},
                    {'config.model_params.tab_nn_config.hidden_size': {'$eq': mp.get('tab_nn_config', {}).get('hidden_size')}},
                    {'config.model_params.tab_nn_config.dropout': {'$eq': mp.get('tab_nn_config', {}).get('dropout')}},
                    {'config.model_params.tab_nn_config.num_layers': {'$eq': mp.get('tab_nn_config', {}).get('num_layers')}},
                    {'config.model_params.tab_nn_config.attn_dropout': {'$eq': mp.get('tab_nn_config', {}).get('attn_dropout')}},
                    {'config.model_params.tab_nn_config.num_heads': {'$eq': mp.get('tab_nn_config', {}).get('num_heads')}},
                    {'config.model_params.tab_nn_config.normalization': {'$eq': mp.get('tab_nn_config', {}).get('normalization')}},
                    {'config.model_params.tab_nn_config.optimizer_config.lr': {'$eq': mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('lr')}},
                    {'config.model_params.tab_nn_config.optimizer_config.weight_decay': {'$eq': mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('weight_decay')}},
                    {'config.model_params.tab_nn_config.optimizer_config.gamma': {'$eq': mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('gamma')}},
                ]
            }
        if db_name is not None and task_name is not None:
            query['config.dbs'] = {
                '$elemMatch': {
                    'db_name': db_name,
                    'task_name': {'$in': [task_name]}
                }
            }
        return list(self.collection.find(query))

    def find_loss_metrics_by_config(self, model_config: Dict[str, Any], tag: str = None) -> list:
        if self.test:
            return []
        mp = model_config.get('model_params', {})
        rdl_config = mp.get('torch_frame_model_cls', {}) == 'resnet'
        if rdl_config:
            query = {
                '$and': [
                    {'config.model_params.encoder_num_layers': {'$eq': mp.get('encoder_num_layers')}},
                    # {'config.model_params.channels': {'$eq': mp.get('channels')}},
                    {'config.model_params.torch_frame_model_cls': {'$eq': mp.get('torch_frame_model_cls')}},
                    {'config.model_params.gnn_config.pre_sf': {'$eq': mp.get('gnn_config', {}).get('pre_sf')}},
                    {'config.model_params.gnn_config.mpnn_type': {'$eq': mp.get('gnn_config', {}).get('mpnn_type')}},
                    # # {'config.model_params.gnn_config.mpnn_design_choice': {'$eq': mp.get('gnn_config', {}).get('mpnn_design_choice')}},
                    # {'config.model_params.gnn_config.post_sf': {'$eq': mp.get('gnn_config', {}).get('post_sf')}},
                    {'config.model_params.gnn_config.hidden_channels': {'$eq': mp.get('gnn_config', {}).get('hidden_channels')}},
                    {'config.model_params.gnn_config.num_heads': {'$eq': mp.get('gnn_config', {}).get('num_heads')}},
                    {'config.model_params.gnn_config.dropout': {'$eq': mp.get('gnn_config', {}).get('dropout')}},
                    {'config.model_params.gnn_config.norm': {'$eq': mp.get('gnn_config', {}).get('norm')}},
                    {'config.model_params.gnn_config.aggregation': {'$eq': mp.get('gnn_config', {}).get('aggregation')}},
                    # {'config.model_params.gnn_config.jk': {'$eq': mp.get('gnn_config', {}).get('jk')}},
                    {'config.model_params.sampler_config.temporal': {'$eq': mp.get('sampler_config', {}).get('temporal')}},
                    {'config.model_params.sampler_config.num_neighbors': {'$eq': mp.get('sampler_config', {}).get('num_neighbors')}},
                    {'config.model_params.sampler_config.num_layers': {'$eq': mp.get('sampler_config', {}).get('num_layers')}},
                    {'config.model_params.sampler_config.loader_type': {'$eq': mp.get('sampler_config', {}).get('loader_type')}},
                    {'config.model_params.optimizer_config.lr': {'$eq': mp.get('optimizer_config', {}).get('lr')}},
                    {'config.model_params.optimizer_config.weight_decay': {'$eq': mp.get('optimizer_config', {}).get('weight_decay')}},
                    {'config.model_params.optimizer_config.scheduler': {'$eq': mp.get('optimizer_config', {}).get('scheduler')}},
                    {'config.model_params.optimizer_config.gamma': {'$eq': mp.get('optimizer_config', {}).get('gamma')}},
                ]
            }
        else:
            query = {
                '$and': [
                    {'config.model_params.dfs_layer': {'$eq': mp.get('dfs_layer')}},
                    {'config.model_params.torch_frame_model_cls': {'$eq': mp.get('torch_frame_model_cls')}},
                    {'config.model_params.batch_size': {'$eq': mp.get('batch_size')}},
                    {'config.model_params.tab_nn_config.hidden_size': {'$eq': mp.get('tab_nn_config', {}).get('hidden_size')}},
                    {'config.model_params.tab_nn_config.dropout': {'$eq': mp.get('tab_nn_config', {}).get('dropout')}},
                    {'config.model_params.tab_nn_config.num_layers': {'$eq': mp.get('tab_nn_config', {}).get('num_layers')}},
                    {'config.model_params.tab_nn_config.attn_dropout': {'$eq': mp.get('tab_nn_config', {}).get('attn_dropout')}},
                    {'config.model_params.tab_nn_config.num_heads': {'$eq': mp.get('tab_nn_config', {}).get('num_heads')}},
                    {'config.model_params.tab_nn_config.normalization': {'$eq': mp.get('tab_nn_config', {}).get('normalization')}},
                    {'config.model_params.tab_nn_config.optimizer_config.lr': {'$eq': mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('lr')}},
                    {'config.model_params.tab_nn_config.optimizer_config.weight_decay': {'$eq': mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('weight_decay')}},
                    {'config.model_params.tab_nn_config.optimizer_config.scheduler': {'$eq': mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('scheduler')}},
                    {'config.model_params.tab_nn_config.optimizer_config.gamma': {'$eq': mp.get('tab_nn_config', {}).get('optimizer_config', {}).get('gamma')}},
                ]
            }
        
        docs = list(self.collection.find(query))
        return docs

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except Exception:  # pragma: no cover
                return obj.hex()
        if _np is not None:
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            try:
                return obj.item()
            except Exception:  # pragma: no cover
                pass
        return str(obj)

    def _config_hash(self, config: Dict[str, Any]) -> str:
        try:
            serialized = json.dumps(config, sort_keys=True, default=self._json_default)
        except TypeError:  # pragma: no cover - fallback
            serialized = json.dumps(self._json_default(config), sort_keys=True, default=self._json_default)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    
