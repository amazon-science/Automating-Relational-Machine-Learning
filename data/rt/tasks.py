"""Task registry for relational transformer experiments.

This module defines predefined task groups that can be referenced from YAML configs.
Tasks are defined as tuples: (database, table, target_column, drop_columns)

Usage in YAML:
  1. Reference a predefined task group:
     train_tasks:
       task_ref: forecast_clf_tasks

  2. Reference multiple groups:
     train_tasks:
       task_ref:
         - forecast_clf_tasks
         - forecast_reg_tasks

  3. Mix references and manual definitions:
     train_tasks:
       task_ref: forecast_clf_tasks
       tasks:
         - db: rel-custom
           table: custom-table
           target: custom_target
           drop: []

  4. Manual definitions only (backward compatible):
     train_tasks:
       - db: rel-f1
         table: driver-dnf
         target: did_not_finish
         drop: []
"""

from typing import Any, Dict, List, Tuple


# tuples are (database, table, target column, leakage columns)

forecast_clf_tasks = [
    ("rel-hm", "user-churn", "churn", []),
    ("rel-stack", "user-badge", "WillGetBadge", []),
    ("rel-stack", "user-engagement", "contribution", []),
    ("rel-avito", "user-visits", "num_click", []),
    ("rel-avito", "user-clicks", "num_click", []),
    ("rel-event", "user-ignore", "target", []),
    ("rel-trial", "study-outcome", "outcome", []),
    ("rel-f1", "driver-dnf", "did_not_finish", []),
    ("rel-event", "user-repeat", "target", []),
    ("rel-f1", "driver-top3", "qualifying", []),
]

# Task tuples with type annotations: (db, table, target, drop_cols, task_type)
# For backward compatibility, task_type is optional and defaults to 'clf'
pt_downstream_test_tasks = [
    ('rel-f1', 'driver-position', 'position', [], 'reg'),  # regression task
    ('rel-trial', 'study-outcome', 'outcome', [], 'clf'),  # classification task
    ('rel-f1', 'driver-dnf', 'did_not_finish', [], 'clf'),
    ('rel-f1', 'driver-top3', 'qualifying', [], 'clf'),
]

pt_in_db_forecast_tasks = [
    ('rel-trial', 'study-adverse', 'adverse', [], 'reg'),
    ('rel-trial', 'site-success', 'success', [], 'reg'),
    ('rel-f1', 'driver-qualifying-beat-teammate', 'beats_teammate', [], 'clf'),
    ('rel-f1', 'driver-will-race', 'will_race', [], 'clf'),
    ('rel-f1', 'driver-podium', 'finished_on_podium', [], 'clf'),
]

pt_in_db_autocomplete_tasks = [
    (
        "rel-f1",
        "results",
        "position",
        [
            "statusId",
            "positionOrder",
            "points",
            "laps",
            "milliseconds",
            "fastestLap",
            "rank",
        ],
    ),
    ("rel-f1", "qualifying", "position", []),
    ("rel-trial", "studies", "enrollment", []),
    ("rel-f1", "constructor_results", "points", []),
    ("rel-f1", "constructor_standings", "position", ["wins", "points"]),
        ("rel-trial", "studies", "has_dmc", []),
    (
        "rel-trial",
        "eligibilities",
        "adult",
        [
            "child",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    (
        "rel-trial",
        "eligibilities",
        "child",
        [
            "adult",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
]

pt_in_domain_tasks = pt_in_db_forecast_tasks + pt_in_db_autocomplete_tasks

pt_out_db_forecast_tasks = [
    ('rel-hm', 'user-churn', 'churn', []),
    ('rel-hm', 'item-sales', 'sales', []),
    ('rel-stack', 'user-badge', 'WillGetBadge', []),
    ('rel-stack', 'user-engagement', 'contribution', []),
    ('rel-stack', 'post-votes', 'popularity', []),
    ('rel-avito', 'user-visits', 'num_click', []),
    ('rel-avito', 'user-clicks', 'num_click', []),
    ('rel-arxiv', 'paper-citation', 'cited', []),
    ('rel-arxiv', 'author-publication', 'publication_count', []),
    ('rel-avito', 'ad-ctr', 'num_click', []),
    # ('rel-ratebeer', 'beer-rating-churn', 'rating_churn', []),
    # ('rel-ratebeer', 'user-rating-churn', 'rating_churn', []),
    # ('rel-ratebeer', 'brewer-dormant', 'dormant', []),
    # ('rel-ratebeer', 'user-rating-count', 'rating_count', [])
]

pt_out_db_autocomplete_tasks = [
    ("rel-avito", "SearchInfo", "IsUserLoggedOn", []),
    ("rel-stack", "postLinks", "LinkTypeId", []),
    ("rel-hm", "transactions", "price", []),
]

pt_out_domain_tasks = pt_out_db_forecast_tasks + pt_out_db_autocomplete_tasks

all_pt_forecast_tasks = pt_out_db_forecast_tasks

all_pt_autocomplete_tasks = pt_in_db_autocomplete_tasks + pt_out_db_autocomplete_tasks

forecast_reg_tasks = [
    ("rel-hm", "item-sales", "sales", []),
    ("rel-stack", "post-votes", "popularity", []),
    ("rel-trial", "site-success", "success_rate", []),
    ("rel-trial", "study-adverse", "num_of_adverse_events", []),
    ("rel-event", "user-attendance", "target", []),
    ("rel-f1", "driver-position", "position", []),
    ("rel-avito", "ad-ctr", "num_click", []),
]

autocomplete_clf_tasks = [
    ("rel-avito", "SearchInfo", "IsUserLoggedOn", []),
    ("rel-stack", "postLinks", "LinkTypeId", []),
    ("rel-trial", "studies", "has_dmc", []),
    (
        "rel-trial",
        "eligibilities",
        "adult",
        [
            "child",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    (
        "rel-trial",
        "eligibilities",
        "child",
        [
            "adult",
            "older_adult",
            "minimum_age",
            "maximum_age",
            "population",
            "criteria",
            "gender_description",
        ],
    ),
    ("rel-event", "event_interest", "not_interested", ["interested"]),
]

autocomplete_reg_tasks = [
    (
        "rel-f1",
        "results",
        "position",
        [
            "statusId",
            "positionOrder",
            "points",
            "laps",
            "milliseconds",
            "fastestLap",
            "rank",
        ],
    ),
    ("rel-f1", "qualifying", "position", []),
    ("rel-trial", "studies", "enrollment", []),
    ("rel-f1", "constructor_results", "points", []),
    ("rel-f1", "constructor_standings", "position", ["wins", "points"]),
    ("rel-hm", "transactions", "price", []),
    ("rel-event", "users", "birthyear", []),
]

all_tasks = (
    forecast_clf_tasks
    + forecast_reg_tasks
    + autocomplete_clf_tasks
    + autocomplete_reg_tasks
)

forecast_tasks = forecast_clf_tasks + forecast_reg_tasks

all_dbs = [
    "rel-hm",
    "rel-stack",
    "rel-avito",
    "rel-event",
    "rel-trial",
    "rel-f1",
]


# Registry mapping names to task lists
TASK_REGISTRY: Dict[str, List[Tuple[str, str, str, List[str]]]] = {
    "forecast_clf_tasks": forecast_clf_tasks,
    "forecast_reg_tasks": forecast_reg_tasks,
    "autocomplete_clf_tasks": autocomplete_clf_tasks,
    "autocomplete_reg_tasks": autocomplete_reg_tasks,
    "all_tasks": all_tasks,
    "forecast_tasks": forecast_tasks,
    "all_pt_forecast_tasks": all_pt_forecast_tasks,
    "all_pt_autocomplete_tasks": all_pt_autocomplete_tasks,
    "pt_in_db_forecast_tasks": pt_in_db_forecast_tasks,
    "pt_in_db_autocomplete_tasks": pt_in_db_autocomplete_tasks,
    "pt_out_db_forecast_tasks": pt_out_db_forecast_tasks,
    "pt_out_db_autocomplete_tasks": pt_out_db_autocomplete_tasks,
    "pt_downstream_test_tasks": pt_downstream_test_tasks,
    "pt_in_domain_tasks": pt_in_domain_tasks,
    "pt_out_domain_tasks": pt_out_domain_tasks,
}


def resolve_task_reference(task_ref: Any) -> List[Tuple[str, str, str, List[str]]]:
    """Resolve a task reference (string or list of strings) to task tuples.

    Args:
        task_ref: Either a string reference to a task group, or a list of such references

    Returns:
        List of task tuples (db, table, target, drop)

    Raises:
        ValueError: If a referenced task group is not found in the registry
    """
    if isinstance(task_ref, str):
        if task_ref not in TASK_REGISTRY:
            available = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise ValueError(
                f"Unknown task reference '{task_ref}'. "
                f"Available references: {available}"
            )
        return list(TASK_REGISTRY[task_ref])

    elif isinstance(task_ref, (list, tuple)):
        # Handle list of references
        all_tasks_resolved = []
        for ref in task_ref:
            if not isinstance(ref, str):
                raise ValueError(
                    f"Task reference must be a string, got {type(ref)}: {ref}"
                )
            if ref not in TASK_REGISTRY:
                available = ", ".join(sorted(TASK_REGISTRY.keys()))
                raise ValueError(
                    f"Unknown task reference '{ref}'. "
                    f"Available references: {available}"
                )
            all_tasks_resolved.extend(TASK_REGISTRY[ref])
        return all_tasks_resolved

    else:
        raise ValueError(
            f"Task reference must be a string or list of strings, got {type(task_ref)}"
        )


def parse_task_config(
    task_config: Any,
    default_split: str = "train"
) -> List[Tuple[str, str, str, str, List[str]]]:
    """Parse task configuration from YAML into standard tuple format.

    Supports three formats:
    1. Direct list of task dicts (backward compatible):
       [{"db": "rel-f1", "table": "driver-dnf", "target": "did_not_finish", "drop": []}]

    2. Task reference:
       {"task_ref": "forecast_clf_tasks"}

    3. Mixed format:
       {"task_ref": "forecast_clf_tasks", "tasks": [{"db": ..., "table": ...}]}

    Args:
        task_config: Configuration from YAML (list or dict)
        default_split: Default split to use if not specified

    Returns:
        List of tuples: (db, table, target, split, drop)
    """
    result = []

    # Handle backward-compatible list format
    if isinstance(task_config, list):
        for task in task_config:
            if isinstance(task, dict):
                result.append((
                    task["db"],
                    task["table"],
                    task["target"],
                    default_split,
                    task.get("drop", []),
                ))
        return result

    # Handle new dict format with references
    if isinstance(task_config, dict):
        # First, resolve any task references
        if "task_ref" in task_config:
            ref_tasks = resolve_task_reference(task_config["task_ref"])
            for task_tuple in ref_tasks:
                # Handle both 4-tuple (db, table, target, drop) and 5-tuple (..., task_type)
                if len(task_tuple) == 5:
                    db, table, target, drop, task_type = task_tuple
                    # For training tasks, we ignore task_type (it's only for eval)
                    result.append((db, table, target, default_split, drop))
                else:
                    db, table, target, drop = task_tuple
                    result.append((db, table, target, default_split, drop))

        # Then add any manually defined tasks
        if "tasks" in task_config:
            manual_tasks = task_config["tasks"]
            if isinstance(manual_tasks, list):
                for task in manual_tasks:
                    if isinstance(task, dict):
                        result.append((
                            task["db"],
                            task["table"],
                            task["target"],
                            default_split,
                            task.get("drop", []),
                        ))

        return result

    # Empty or None config
    if task_config is None:
        return []

    raise ValueError(
        f"Invalid task configuration format. Expected list or dict, got {type(task_config)}"
    )
