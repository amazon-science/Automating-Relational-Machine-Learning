import json
import pkgutil
from collections import defaultdict
from functools import lru_cache
from typing import List
import os
import pooch

from relbench.base import AutoCompleteTask, BaseTask, TaskType
from relbench.datasets import get_dataset, on_disks, DEFAULT_CACHE_DIR
from relbench.tasks import avito, event, f1, hm, ieeecis, stack, trial, arxiv, fakef1

task_registry = defaultdict(dict)

hashes_str = pkgutil.get_data(__name__, "hashes.json")
hashes = json.loads(hashes_str)

DOWNLOAD_REGISTRY = pooch.create(
    path=os.environ.get("XDG_CACHE_HOME") or DEFAULT_CACHE_DIR,
    base_url="https://relbench.stanford.edu/download/",
    registry=hashes,
)

DOWNLOADABLE_TASKS = [
    "driver-dnf",
    "driver-top3",
    "driver-position",
    "user-churn",
    "item-sales",
    "study-outcome",
    "study-adverse",
    "site-success",
    "condition-sponsor-run",
    "site-sponsor-run",
    "user-engagement",
    "user-badge",
    "post-votes",
    "user-post-comment",
    "post-post-related",
    "user-repeat",
    "user-ignore",
    "user-attendance",
    "user-visits",
    "user-clicks",
    "ad-ctr",
    "user-ad-visit"
]


def register_task(
    dataset_name: str,
    task_name: str,
    cls: BaseTask,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`BaseTask` subclass with the given name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        cls: The class of the task.
        args: The arguments to instantiate the task.
        kwargs: The keyword arguments to instantiate the task.

    The name is used to enable caching and downloading functionalities.
    `cache_dir` is added to kwargs by default. If you want to override it, you
    can pass `cache_dir` as a keyword argument in `kwargs`.
    """

    cache_dir = f"{pooch.os_cache('relbench')}/{dataset_name}/tasks/{task_name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str) -> List[str]:
    r"""Return a list of names of the registered tasks for the given dataset."""
    return list(task_registry[dataset_name].keys())


def download_task(dataset_name: str, task_name: str) -> None:
    r"""Download task from RelBench server into its cache directory.

    The downloaded task tables will be automatically picked up by the task object, when
    `task.get_table(split)` is called.
    """

    DOWNLOAD_REGISTRY.fetch(
        f"{dataset_name}/tasks/{task_name}.zip",
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True,
    )


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str, download=False) -> BaseTask:
    r"""Return a task object by name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        download: If True, download the task from the RelBench server.

    Returns:
        BaseTask: The task object.

    If `download` is True, the task tables (train, val, test) comprising the
    task will be downloaded into the cache from the RelBench server. If you use
    `download=False` the first time, the task tables will be computed from
    scratch using the database.

    Once the task tables are cached, either because of download or computing from
    scratch, the cache will be used. `download=True` will verify that the
    cached task tables matches the RelBench version even in this case.
    """

    if download and task_name in DOWNLOADABLE_TASKS:
        download_task(dataset_name, task_name)
    dataset = get_dataset(dataset_name, download = True if dataset_name not in on_disks else False)
    cls, args, kwargs = task_registry[dataset_name][task_name]
    task = cls(dataset, *args, **kwargs)
    return task


register_task("rel-avito", "ad-ctr", avito.AdCTRTask)
register_task("rel-avito", "user-visits", avito.UserVisitsTask)
register_task("rel-avito", "user-clicks", avito.UserClicksTask)
register_task("rel-avito", "user-ad-visit", avito.UserAdVisitTask)
register_task(
    "rel-avito",
    "searchstream-click",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="SearchStream",
    target_col="IsClick",
)
register_task(
    "rel-avito",
    "searchinfo-isuserloggedon",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="SearchInfo",
    target_col="IsUserLoggedOn",
)

register_task("rel-event", "user-attendance", event.UserAttendanceTask)
register_task("rel-event", "user-repeat", event.UserRepeatTask)
register_task("rel-event", "user-ignore", event.UserIgnoreTask)
# register_task(
#     "rel-event",
#     "event_interest-iterested",
#     AutoCompleteTask,
#     task_type=TaskType.BINARY_CLASSIFICATION,
#     entity_table="event_interest",
#     target_col="interested",
#     remove_columns=[
#         ("event_interest", "not_interested"),
#     ],
# )

register_task("rel-f1", "driver-position", f1.DriverPositionTask)
register_task("rel-f1", "driver-dnf", f1.DriverDNFTask)
register_task("rel-f1", "driver-top3", f1.DriverTop3Task)
register_task("rel-f1", "driver-podium", f1.DriverPodiumTask)
register_task("rel-f1", "driver-scores-points", f1.DriverScoresPointsTask)
register_task("rel-f1", "constructor-scores-points", f1.ConstructorScoresPointsTask)
register_task("rel-f1", "constructor-points", f1.ConstructorPointsTask)
register_task("rel-f1", "driver-wins", f1.DriverWinsTask)
register_task("rel-f1", "driver-position-change", f1.DriverPositionChangeTask)
register_task("rel-f1", "circuit-fastest-lap", f1.CircuitFastestLapTask)
# New binary classification tasks
register_task("rel-f1", "driver-will-race", f1.DriverWillRaceTask)
register_task("rel-f1", "driver-qualifying-beat-teammate", f1.DriverQualifyingBeatTeammateTask)
register_task("rel-f1", "constructor-podium", f1.ConstructorPodiumTask)
register_task("rel-f1", "driver-improved-standings", f1.DriverImprovedStandingsTask)
register_task("rel-f1", "constructor-win", f1.ConstructorWinTask)
register_task("rel-f1", "driver-fastest-lap", f1.DriverFastestLapTask)
register_task("rel-f1", "driver-consistent-finisher", f1.DriverConsistentFinisherTask)
register_task("rel-f1", "driver-race-compete", f1.DriverRaceCompeteTask)
# AutoComplete tasks
register_task(
    "rel-f1",
    "results-position",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="results",
    target_col="position",
    remove_columns=[
        ("results", "statusId"),
        ("results", "positionOrder"),
        ("results", "points"),
        ("results", "laps"),
        ("results", "milliseconds"),
        ("results", "fastestLap"),
        ("results", "rank"),
    ],
)
# Needs > 10 epochs
register_task(
    "rel-f1",
    "qualifying-position",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="qualifying",
    target_col="position",
    remove_columns=[],
)


register_task("rel-hm", "user-item-purchase", hm.UserItemPurchaseTask)
register_task("rel-hm", "user-churn", hm.UserChurnTask)
register_task("rel-hm", "item-sales", hm.ItemSalesTask)
register_task("rel-hm", "customer-spending", hm.CustomerSpendingTask)
register_task('rel-hm', 'customer-average-price', hm.CustomerAvgPriceTask)



register_task("rel-stack", "user-engagement", stack.UserEngagementTask)
register_task("rel-stack", "post-votes", stack.PostVotesTask)
register_task("rel-stack", "user-badge", stack.UserBadgeTask)
register_task("rel-stack", "user-post-comment", stack.UserPostCommentTask)
register_task("rel-stack", "user-comment-count", stack.UserCommentCountTask)
register_task("rel-stack", "post-post-related", stack.PostPostRelatedTask)
register_task(
    "rel-stack",
    "badges-class",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="badges",
    target_col="Class",
    remove_columns=[("badges", "TagBased"), ("badges", "Name")],
)

register_task("rel-trial", "study-outcome", trial.StudyOutcomeTask)
register_task("rel-trial", "study-adverse", trial.StudyAdverseTask)
register_task("rel-trial", "site-success", trial.SiteSuccessTask)
register_task("rel-trial", "condition-sponsor-run", trial.ConditionSponsorRunTask)
register_task("rel-trial", "site-sponsor-run", trial.SiteSponsorRunTask)

# IEEE-CIS Fraud Detection Tasks
register_task(
    "rel-ieeecis",
    "transaction-fraud",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="transactions",
    target_col="isFraud",
    remove_columns=[("transactions", f"V{i}") for i in range(1, 340)] + [("transactions", "index")]
)

register_task(
    "rel-avs",
    "repeater",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="history",
    target_col="repeater",
    remove_columns=[("history", "repeattrips")]
)

register_task("rel-arxiv", "paper-citation", arxiv.PaperCitationTask)
register_task("rel-arxiv", "author-category", arxiv.AuthorCategoryTask)
register_task("rel-arxiv", "author-publication", arxiv.AuthorPublicationTask)
register_task("rel-arxiv", "co-citation", arxiv.CoCitationTask)

register_task("rel-fakef1", "driver-dnf-fake", fakef1.DriverDNFTask)


