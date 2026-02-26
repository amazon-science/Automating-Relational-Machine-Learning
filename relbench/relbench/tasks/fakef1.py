import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    roc_auc,
)



class DriverDNFTask(EntityTask):
    r"""Predict the if each driver will DNF (not finish) a race in the next 1 month."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "did_not_finish"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        results = db.table_dict["results"].df
        drivers = db.table_dict["drivers"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) AS did_not_finish
                FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date  > t.timestamp
                LEFT JOIN
                    drivers dri
                ON
                    re.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM results
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId

            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )