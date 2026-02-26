import pandas as pd

def make_feature_importance_df(
    main_metric: float,
    shuffled_metrics: dict[str, float]
):
    """
    main_metric :
        The evaluation metric
    shuffled_metrics :
        The evaluation metric after shuffling the values in one column.
    """
    shuffled_metrics = pd.Series(shuffled_metrics)
    shuffled_metrics.name = 'metric'
    shuffled_metrics.index.name = 'column_name'
    shuffled_metrics = shuffled_metrics.reset_index()
    shuffled_metrics['importance'] = main_metric - shuffled_metrics['metric']
    shuffled_metrics = shuffled_metrics.sort_values('importance', ascending=False)
    return shuffled_metrics