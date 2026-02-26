import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)


class DriverPositionTask(EntityTask):
    r"""Predict the average finishing position of each driver all races in the next 2
    months."""

    task_type = TaskType.REGRESSION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "position"
    timedelta = pd.Timedelta(days=60)
    metrics = [r2, mae, rmse]
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
                    mean(re.positionOrder) as position,
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


class DriverTop3Task(EntityTask):
    r"""Predict if each driver will qualify in the top-3 for a race within the next 1
    month."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "qualifying"
    timedelta = pd.Timedelta(days=30)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        qualifying = db.table_dict["qualifying"].df
        drivers = db.table_dict["drivers"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    dri.driverId as driverId,
                    CASE
                        WHEN MIN(qu.position) <= 3 THEN 1
                        ELSE 0
                    END AS qualifying
                FROM
                    timestamp_df t
                LEFT JOIN
                    qualifying qu
                ON
                    qu.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and qu.date > t.timestamp
                LEFT JOIN
                    drivers dri
                ON
                    qu.driverId = dri.driverId
                WHERE
                    dri.driverId IN (
                        SELECT DISTINCT driverId
                        FROM qualifying
                        WHERE date > t.timestamp - INTERVAL '1 year'
                    )
                GROUP BY t.timestamp, dri.driverId

            ;
            """
        ).df()

        df["qualifying"] = df["qualifying"].astype("int64")

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverPodiumTask(EntityTask):
    r"""Predict whether each driver will finish on the podium (top-3)
    in at least one race within the next 3 months."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "podium"
    timedelta = pd.Timedelta(days=90)  # 3 months
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                re.driverId AS driverId,
                CASE WHEN MIN(re.positionOrder) <= 3 THEN 1 ELSE 0 END AS podium
            FROM timestamp_df t
            LEFT JOIN results re
              ON re.date <= t.timestamp + INTERVAL '{self.timedelta}'
             AND re.date  > t.timestamp
            WHERE re.driverId IN (
                SELECT DISTINCT driverId
                FROM results
                WHERE date > t.timestamp - INTERVAL '1 year'
            )
            GROUP BY t.timestamp, re.driverId;
            """
        ).df()

        df["podium"] = df["podium"].astype("int64")
        return Table(df=df,
                     fkey_col_to_pkey_table={self.entity_col: self.entity_table},
                     pkey_col=None,
                     time_col=self.time_col)


class DriverScoresPointsTask(EntityTask):
    r"""Predict whether each driver will score championship points (>0)
    in any race within the next 3 months."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "scored_points"
    timedelta = pd.Timedelta(days=90)  # 3 months
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                re.driverId AS driverId,
                MAX(CASE WHEN re.points > 0 THEN 1 ELSE 0 END) AS scored_points
            FROM timestamp_df t
            LEFT JOIN results re
              ON re.date <= t.timestamp + INTERVAL '{self.timedelta}'
             AND re.date  > t.timestamp
            WHERE re.driverId IN (
                SELECT DISTINCT driverId
                FROM results
                WHERE date > t.timestamp - INTERVAL '1 year'
            )
            GROUP BY t.timestamp, re.driverId;
            """
        ).df()

        df["scored_points"] = df["scored_points"].astype("int64")
        return Table(df=df,
                     fkey_col_to_pkey_table={self.entity_col: self.entity_table},
                     pkey_col=None,
                     time_col=self.time_col)


class ConstructorScoresPointsTask(EntityTask):
    r"""Predict whether each constructor will score any points (>0 in constructor_results)
    within the next 3 months."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "constructorId"
    entity_table = "constructors"
    time_col = "date"
    target_col = "scored_points"
    timedelta = pd.Timedelta(days=90)  # 3 months
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        constructor_results = db.table_dict["constructor_results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                cr.constructorId AS constructorId,
                MAX(CASE WHEN cr.points > 0 THEN 1 ELSE 0 END) AS scored_points
            FROM timestamp_df t
            LEFT JOIN constructor_results cr
              ON cr.date <= t.timestamp + INTERVAL '{self.timedelta}'
             AND cr.date  > t.timestamp
            WHERE cr.constructorId IN (
                SELECT DISTINCT constructorId
                FROM constructor_results
                WHERE date > t.timestamp - INTERVAL '1 year'
            )
            GROUP BY t.timestamp, cr.constructorId;
            """
        ).df()

        df["scored_points"] = df["scored_points"].astype("int64")
        return Table(df=df,
                     fkey_col_to_pkey_table={self.entity_col: self.entity_table},
                     pkey_col=None,
                     time_col=self.time_col)


class ConstructorPointsTask(EntityTask):
    r"""Predict the total points each constructor will score in all races over the
    next 90 days."""

    task_type = TaskType.REGRESSION
    entity_col = "constructorId"
    entity_table = "constructors"
    time_col = "date"
    target_col = "points"
    timedelta = pd.Timedelta(days=90)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        constructor_results = db.table_dict["constructor_results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                cr.constructorId AS constructorId,
                SUM(cr.points) AS points
            FROM
                timestamp_df t
            LEFT JOIN
                constructor_results cr
            ON
                cr.date > t.timestamp AND
                cr.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                -- Only predict for constructors active in the year prior
                cr.constructorId IN (
                    SELECT DISTINCT constructorId
                    FROM constructor_results
                    WHERE date BETWEEN t.timestamp - INTERVAL '1 year' AND t.timestamp
                )
            GROUP BY
                t.timestamp, cr.constructorId
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class DriverWinsTask(EntityTask):
    r"""Predict the number of races each driver will win (i.e., finish in
    position 1) over the next year."""

    task_type = TaskType.REGRESSION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "wins"
    timedelta = pd.Timedelta(days=365)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 15 # Reduced due to longer timedelta

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                r.driverId AS driverId,
                COUNT(*) AS wins
            FROM
                timestamp_df t
            LEFT JOIN
                results r
            ON
                r.date > t.timestamp AND
                r.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                r.positionOrder = 1 AND
                -- Only predict for drivers who were active recently
                r.driverId IN (
                    SELECT DISTINCT driverId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                )
            GROUP BY
                t.timestamp, r.driverId
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class DriverPositionChangeTask(EntityTask):
    r"""Predict the average change between a driver's starting grid position
    and their final position for all races in the next 4 months. A negative
    value indicates gaining positions."""

    task_type = TaskType.REGRESSION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "position_change"
    timedelta = pd.Timedelta(days=120)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                r.driverId AS driverId,
                AVG(r.positionOrder - r.grid) AS position_change
            FROM
                timestamp_df t
            LEFT JOIN
                results r
            ON
                r.date > t.timestamp AND
                r.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                r.grid > 0 AND -- Exclude pit lane starts that skew data
                r.driverId IN (
                    SELECT DISTINCT driverId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                )
            GROUP BY
                t.timestamp, r.driverId
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class CircuitFastestLapTask(EntityTask):
    r"""Predict the minimum fastest lap time (in milliseconds) across all
    drivers for the next race at a given circuit within the next 2 years."""

    task_type = TaskType.REGRESSION
    entity_col = "circuitId"
    entity_table = "circuits"
    time_col = "date"
    target_col = "fastest_lap_ms"
    timedelta = pd.Timedelta(days=730)
    metrics = [r2, mae, rmse]
    num_eval_timestamps = 10 # Reduced due to very long timedelta

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
            WITH future_races AS (
                SELECT
                    t.timestamp,
                    ra.circuitId,
                    -- Find the earliest race at each circuit after the timestamp
                    MIN(ra.date) as next_race_date
                FROM
                    timestamp_df t
                JOIN
                    races ra
                ON
                    ra.date > t.timestamp AND
                    ra.date <= t.timestamp + INTERVAL '{self.timedelta}'
                GROUP BY t.timestamp, ra.circuitId
            )
            SELECT
                fr.timestamp AS date,
                fr.circuitId AS circuitId,
                MIN(re.milliseconds) AS fastest_lap_ms
            FROM
                future_races fr
            JOIN
                results re ON fr.next_race_date = re.date
            WHERE
                re.milliseconds IS NOT NULL
            GROUP BY
                fr.timestamp, fr.circuitId
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverWillRaceTask(EntityTask):
    r"""Predict whether each driver will participate in any race within the next
    2 months. Useful for predicting driver contract renewals and retirements."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "will_race"
    timedelta = pd.Timedelta(days=60)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp as date,
                dri.driverId as driverId,
                CASE
                    WHEN COUNT(re.resultId) > 0 THEN 1
                    ELSE 0
                END AS will_race
            FROM
                timestamp_df t
            CROSS JOIN
                (SELECT DISTINCT driverId FROM results) dri
            LEFT JOIN
                results re
            ON
                re.driverId = dri.driverId
                AND re.date > t.timestamp
                AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                -- Only consider drivers who were active in the past year
                dri.driverId IN (
                    SELECT DISTINCT driverId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                      AND date <= t.timestamp
                )
            GROUP BY t.timestamp, dri.driverId
            """
        ).df()

        df["will_race"] = df["will_race"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverQualifyingBeatTeammateTask(EntityTask):
    r"""Predict whether a driver will out-qualify their teammate (same constructor)
    in the majority of races within the next 3 months."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "beats_teammate"
    timedelta = pd.Timedelta(days=90)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        qualifying = db.table_dict["qualifying"].df
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            WITH driver_constructor AS (
                -- Get each driver's constructor for each race in the future window
                SELECT DISTINCT
                    t.timestamp,
                    re.driverId,
                    re.constructorId,
                    re.raceId
                FROM timestamp_df t
                JOIN results re
                  ON re.date > t.timestamp
                 AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
            ),
            quali_comparison AS (
                SELECT
                    dc.timestamp,
                    dc.driverId,
                    dc.raceId,
                    q1.position as driver_position,
                    MIN(q2.position) as best_teammate_position
                FROM driver_constructor dc
                JOIN qualifying q1
                  ON dc.driverId = q1.driverId
                 AND dc.raceId = q1.raceId
                LEFT JOIN qualifying q2
                  ON dc.raceId = q2.raceId
                 AND q2.driverId != dc.driverId
                 AND q2.constructorId = dc.constructorId
                WHERE q1.position IS NOT NULL
                GROUP BY dc.timestamp, dc.driverId, dc.raceId, q1.position
            )
            SELECT
                qc.timestamp as date,
                qc.driverId,
                CASE
                    WHEN SUM(CASE WHEN driver_position < best_teammate_position THEN 1 ELSE 0 END)
                         > COUNT(*) * 0.5
                    THEN 1
                    ELSE 0
                END AS beats_teammate
            FROM quali_comparison qc
            WHERE qc.best_teammate_position IS NOT NULL
            GROUP BY qc.timestamp, qc.driverId
            HAVING COUNT(*) >= 2  -- Need at least 2 races for comparison
            """
        ).df()

        df["beats_teammate"] = df["beats_teammate"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ConstructorPodiumTask(EntityTask):
    r"""Predict whether a constructor will achieve at least one podium finish
    (top-3) in any race within the next 3 months."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "constructorId"
    entity_table = "constructors"
    time_col = "date"
    target_col = "podium"
    timedelta = pd.Timedelta(days=90)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                re.constructorId AS constructorId,
                CASE
                    WHEN MIN(re.positionOrder) <= 3 THEN 1
                    ELSE 0
                END AS podium
            FROM timestamp_df t
            LEFT JOIN results re
              ON re.date > t.timestamp
             AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE re.constructorId IN (
                SELECT DISTINCT constructorId
                FROM results
                WHERE date > t.timestamp - INTERVAL '1 year'
                  AND date <= t.timestamp
            )
            GROUP BY t.timestamp, re.constructorId
            """
        ).df()

        df["podium"] = df["podium"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverImprovedStandingsTask(EntityTask):
    r"""Predict whether a driver will improve their championship standing
    (lower position number is better) over the next 4 months compared to current."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "improved_standing"
    timedelta = pd.Timedelta(days=120)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        standings = db.table_dict["standings"].df

        df = duckdb.sql(
            f"""
            WITH current_standings AS (
                SELECT
                    t.timestamp,
                    s.driverId,
                    s.position as current_position
                FROM timestamp_df t
                LEFT JOIN standings s
                  ON s.date <= t.timestamp
                WHERE s.date = (
                    SELECT MAX(date)
                    FROM standings
                    WHERE driverId = s.driverId
                      AND date <= t.timestamp
                )
            ),
            future_standings AS (
                SELECT
                    t.timestamp,
                    s.driverId,
                    MIN(s.position) as best_future_position
                FROM timestamp_df t
                LEFT JOIN standings s
                  ON s.date > t.timestamp
                 AND s.date <= t.timestamp + INTERVAL '{self.timedelta}'
                GROUP BY t.timestamp, s.driverId
            )
            SELECT
                cs.timestamp as date,
                cs.driverId,
                CASE
                    WHEN fs.best_future_position < cs.current_position THEN 1
                    ELSE 0
                END AS improved_standing
            FROM current_standings cs
            JOIN future_standings fs
              ON cs.timestamp = fs.timestamp
             AND cs.driverId = fs.driverId
            WHERE cs.current_position IS NOT NULL
              AND fs.best_future_position IS NOT NULL
            """
        ).df()

        df["improved_standing"] = df["improved_standing"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ConstructorWinTask(EntityTask):
    r"""Predict whether a constructor will win at least one race (driver finishes 1st)
    within the next 3 months."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "constructorId"
    entity_table = "constructors"
    time_col = "date"
    target_col = "will_win"
    timedelta = pd.Timedelta(days=90)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp AS date,
                re.constructorId AS constructorId,
                MAX(CASE WHEN re.positionOrder = 1 THEN 1 ELSE 0 END) AS will_win
            FROM timestamp_df t
            LEFT JOIN results re
              ON re.date > t.timestamp
             AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE re.constructorId IN (
                SELECT DISTINCT constructorId
                FROM results
                WHERE date > t.timestamp - INTERVAL '1 year'
                  AND date <= t.timestamp
            )
            GROUP BY t.timestamp, re.constructorId
            """
        ).df()

        df["will_win"] = df["will_win"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverFastestLapTask(EntityTask):
    r"""Predict whether a driver will achieve at least one fastest lap
    in any race within the next 2 months."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "fastest_lap"
    timedelta = pd.Timedelta(days=60)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp as date,
                dri.driverId as driverId,
                CASE
                    WHEN COUNT(CASE WHEN re.rank = 1 THEN 1 END) > 0 THEN 1
                    ELSE 0
                END AS fastest_lap
            FROM
                timestamp_df t
            CROSS JOIN
                (SELECT DISTINCT driverId FROM results) dri
            LEFT JOIN
                results re
            ON
                re.driverId = dri.driverId
                AND re.date > t.timestamp
                AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                dri.driverId IN (
                    SELECT DISTINCT driverId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                      AND date <= t.timestamp
                )
            GROUP BY t.timestamp, dri.driverId
            """
        ).df()

        df["fastest_lap"] = df["fastest_lap"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class DriverConsistentFinisherTask(EntityTask):
    r"""Predict whether a driver will finish in the points (top 10) in more than
    70% of races within the next 3 months. Indicates consistency."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "consistent_finisher"
    timedelta = pd.Timedelta(days=90)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp as date,
                re.driverId as driverId,
                CASE
                    WHEN SUM(CASE WHEN re.positionOrder <= 10 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) > 0.7
                    THEN 1
                    ELSE 0
                END AS consistent_finisher
            FROM
                timestamp_df t
            LEFT JOIN
                results re
            ON
                re.date > t.timestamp
                AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                re.driverId IN (
                    SELECT DISTINCT driverId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                      AND date <= t.timestamp
                )
                AND re.positionOrder IS NOT NULL
            GROUP BY t.timestamp, re.driverId
            HAVING COUNT(*) >= 3  -- Need at least 3 races for consistency measure
            """
        ).df()

        df["consistent_finisher"] = df["consistent_finisher"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class ConstructorDominantTask(EntityTask):
    r"""Predict whether a constructor will achieve 'dominant' performance in the next
    3 months, defined as winning >50% of races AND achieving >50% podium rate."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "constructorId"
    entity_table = "constructors"
    time_col = "date"
    target_col = "dominant"
    timedelta = pd.Timedelta(days=90)
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
            WITH race_counts AS (
                SELECT
                    t.timestamp,
                    COUNT(DISTINCT ra.raceId) as total_races
                FROM timestamp_df t
                LEFT JOIN races ra
                  ON ra.date > t.timestamp
                 AND ra.date <= t.timestamp + INTERVAL '{self.timedelta}'
                GROUP BY t.timestamp
            ),
            constructor_performance AS (
                SELECT
                    t.timestamp,
                    re.constructorId,
                    COUNT(DISTINCT re.raceId) as races_participated,
                    SUM(CASE WHEN re.positionOrder = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN re.positionOrder <= 3 THEN 1 ELSE 0 END) as podiums
                FROM timestamp_df t
                LEFT JOIN results re
                  ON re.date > t.timestamp
                 AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                WHERE re.constructorId IN (
                    SELECT DISTINCT constructorId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                      AND date <= t.timestamp
                )
                GROUP BY t.timestamp, re.constructorId
            )
            SELECT
                cp.timestamp as date,
                cp.constructorId,
                CASE
                    WHEN cp.wins * 1.0 / rc.total_races > 0.5 
                     AND cp.podiums * 1.0 / cp.races_participated > 0.5
                    THEN 1
                    ELSE 0
                END AS dominant
            FROM constructor_performance cp
            JOIN race_counts rc ON cp.timestamp = rc.timestamp
            WHERE cp.races_participated >= 3  -- Need participation in at least 3 races
              AND rc.total_races >= 3
            """
        ).df()

        df["dominant"] = df["dominant"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class DriverPerformanceTierTask(EntityTask):
    r"""Predict which performance tier a driver will fall into over the next 4 months:
    0 = Elite (avg position <= 5 and at least 1 podium)
    1 = Midfield (avg position 6-12)
    2 = Backmarker (avg position > 12)"""

    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_col = "driverId"
    entity_table = "drivers"
    time_col = "date"
    target_col = "performance_tier"
    timedelta = pd.Timedelta(days=120)
    metrics = [accuracy, f1]  # f1 will use macro averaging for multiclass
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
            WITH driver_stats AS (
                SELECT
                    t.timestamp,
                    re.driverId,
                    AVG(re.positionOrder) as avg_position,
                    MIN(re.positionOrder) as best_position,
                    COUNT(*) as race_count
                FROM timestamp_df t
                LEFT JOIN results re
                  ON re.date > t.timestamp
                 AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                WHERE re.driverId IN (
                    SELECT DISTINCT driverId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                      AND date <= t.timestamp
                )
                AND re.positionOrder IS NOT NULL
                GROUP BY t.timestamp, re.driverId
            )
            SELECT
                timestamp as date,
                driverId,
                CASE
                    WHEN avg_position <= 5 AND best_position <= 3 THEN 0
                    WHEN avg_position <= 12 THEN 1
                    ELSE 2
                END AS performance_tier
            FROM driver_stats
            WHERE race_count >= 3  -- Need at least 3 races for reliable classification
            """
        ).df()

        df["performance_tier"] = df["performance_tier"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class ConstructorCompetitivenessTask(EntityTask):
    r"""Predict the competitiveness class of each constructor over the next 3 months:
    0 = Championship Contender (top 3 in points, multiple wins)
    1 = Regular Podium Finisher (4-6 podiums, no championship threat)
    2 = Points Scorer (regular points, rare podiums)
    3 = Struggling (minimal points, no podiums)"""

    task_type = TaskType.MULTICLASS_CLASSIFICATION
    entity_col = "constructorId"
    entity_table = "constructors"
    time_col = "date"
    target_col = "competitiveness_class"
    timedelta = pd.Timedelta(days=90)
    metrics = [accuracy, f1]
    num_eval_timestamps = 40

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df
        constructor_results = db.table_dict["constructor_results"].df

        df = duckdb.sql(
            f"""
            WITH constructor_stats AS (
                SELECT
                    t.timestamp,
                    re.constructorId,
                    SUM(re.points) as total_points,
                    SUM(CASE WHEN re.positionOrder = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN re.positionOrder <= 3 THEN 1 ELSE 0 END) as podiums,
                    COUNT(DISTINCT re.raceId) as races_count
                FROM timestamp_df t
                LEFT JOIN results re
                  ON re.date > t.timestamp
                 AND re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                WHERE re.constructorId IN (
                    SELECT DISTINCT constructorId
                    FROM results
                    WHERE date > t.timestamp - INTERVAL '1 year'
                      AND date <= t.timestamp
                )
                GROUP BY t.timestamp, re.constructorId
            ),
            ranked_constructors AS (
                SELECT
                    timestamp,
                    constructorId,
                    total_points,
                    wins,
                    podiums,
                    races_count,
                    RANK() OVER (PARTITION BY timestamp ORDER BY total_points DESC) as points_rank
                FROM constructor_stats
            )
            SELECT
                timestamp as date,
                constructorId,
                CASE
                    WHEN points_rank <= 3 AND wins >= 2 THEN 0
                    WHEN podiums >= 4 THEN 1
                    WHEN total_points > (races_count * 2) THEN 2
                    ELSE 3
                END AS competitiveness_class
            FROM ranked_constructors
            WHERE races_count >= 3  -- Need at least 3 races
            """
        ).df()

        df["competitiveness_class"] = df["competitiveness_class"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class RaceWillHaveSafetyCarTask(EntityTask):
    r"""Predict whether a race will have at least one safety car deployment.
    This is indicated by lap times significantly slower than normal racing pace
    or status codes indicating safety car periods."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "raceId"
    entity_table = "races"
    time_col = "date"
    target_col = "has_safety_car"
    timedelta = pd.Timedelta(days=0)  # Predict for the race on this date
    metrics = [average_precision, accuracy, f1, roc_auc]
    num_eval_timestamps = 60

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df
        races = db.table_dict["races"].df

        df = duckdb.sql(
            f"""
            WITH race_incidents AS (
                SELECT
                    ra.raceId,
                    ra.date,
                    -- Count DNFs and accidents as proxy for safety car likelihood
                    COUNT(CASE WHEN re.statusId NOT IN (1, 11, 12) THEN 1 END) as incidents,
                    -- Check for multiple retirements (often indicates safety car)
                    COUNT(CASE WHEN re.positionOrder IS NULL THEN 1 END) as retirements
                FROM races ra
                LEFT JOIN results re ON ra.raceId = re.raceId
                GROUP BY ra.raceId, ra.date
            )
            SELECT
                t.timestamp as date,
                ri.raceId,
                CASE
                    -- If 3+ incidents or 4+ retirements, likely had safety car
                    WHEN ri.incidents >= 3 OR ri.retirements >= 4 THEN 1
                    ELSE 0
                END AS has_safety_car
            FROM timestamp_df t
            JOIN races ra ON ra.date = t.timestamp
            JOIN race_incidents ri ON ra.raceId = ri.raceId
            WHERE ra.circuitId IN (
                -- Only predict for circuits that have hosted races recently
                SELECT DISTINCT circuitId
                FROM races
                WHERE date > t.timestamp - INTERVAL '3 years'
                  AND date < t.timestamp
            )
            """
        ).df()

        df["has_safety_car"] = df["has_safety_car"].astype("int64")
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )

class DriverRaceCompeteTask(RecommendationTask):
    r"""Predict in which races a driver will compete in the next 1 year."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "driverId"
    src_entity_table = "drivers"
    dst_entity_col = "raceId"
    dst_entity_table = "races"
    target_col = "raceId"
    time_col = "date"
    timedelta = pd.Timedelta(days=365)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        results = db.table_dict["results"].df

        df = duckdb.sql(
            f"""
                SELECT
                    t.timestamp as date,
                    re.driverId as driverId,
                    LIST(DISTINCT re.raceId) as raceId
                    FROM
                    timestamp_df t
                LEFT JOIN
                    results re
                ON
                    re.date <= t.timestamp + INTERVAL '{self.timedelta}'
                    and re.date > t.timestamp
                GROUP BY t.timestamp, re.driverId
            ;
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )