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


class UserItemPurchaseTask(RecommendationTask):
    r"""Predict the list of articles each customer will purchase in the next seven
    days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "customer_id"
    src_entity_table = "customer"
    dst_entity_col = "article_id"
    dst_entity_table = "article"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=7)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]
    eval_k = 12

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                transactions.customer_id,
                LIST(DISTINCT transactions.article_id) AS article_id
            FROM
                timestamp_df t
            LEFT JOIN
                transactions
            ON
                transactions.t_dat > t.timestamp AND
                transactions.t_dat <= t.timestamp + INTERVAL '{self.timedelta} days'
            GROUP BY
                t.timestamp,
                transactions.customer_id
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


class UserChurnTask(EntityTask):
    r"""Predict the churn for a customer (no transactions) in the next week."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=7)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        customer = db.table_dict["customer"].df
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                customer_id,
                CAST(
                    NOT EXISTS (
                        SELECT 1
                        FROM transactions
                        WHERE
                            transactions.customer_id = customer.customer_id AND
                            t_dat > timestamp AND
                            t_dat <= timestamp + INTERVAL '{self.timedelta}'
                    ) AS INTEGER
                ) AS churn
            FROM
                timestamp_df,
                customer,
            WHERE
                EXISTS (
                    SELECT 1
                    FROM transactions
                    WHERE
                        transactions.customer_id = customer.customer_id AND
                        t_dat > timestamp - INTERVAL '{self.timedelta}' AND
                        t_dat <= timestamp
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class ItemSalesTask(EntityTask):
    r"""Predict the total sales for an article (the sum of prices of the associated
    transactions) in the next week."""

    task_type = TaskType.REGRESSION
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = "sales"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        article = db.table_dict["article"].df

        df = duckdb.sql(
            f"""
            SELECT
                timestamp,
                article_id,
                sales
            FROM
                timestamp_df,
                article,
                (
                    SELECT
                        COALESCE(SUM(price), 0) as sales
                    FROM
                        transactions,
                    WHERE
                        transactions.article_id = article.article_id AND
                        t_dat > timestamp AND
                        t_dat <= timestamp + INTERVAL '{self.timedelta}'
                )
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"article_id": "article"},
            pkey_col=None,
            time_col="timestamp",
        )


class CustomerSpendingTask(EntityTask):
    r"""Predict the total spending of each customer in the next 7 days."""

    task_type = TaskType.REGRESSION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "total_spending"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                tr.customer_id,
                SUM(tr.price) AS total_spending
            FROM
                timestamp_df t
            LEFT JOIN
                transactions tr
            ON
                tr.t_dat > t.timestamp AND
                tr.t_dat <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                -- Only predict for customers who were active in the last 6 months
                tr.customer_id IN (
                    SELECT DISTINCT customer_id
                    FROM transactions
                    WHERE t_dat BETWEEN t.timestamp - INTERVAL '6 months' AND t.timestamp
                )
            GROUP BY
                t.timestamp, tr.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="timestamp",
        )

class ItemTransactionCountTask(EntityTask):
    r"""Predict the total number of transactions (units sold) for an article
    in the next 7 days."""

    task_type = TaskType.REGRESSION
    entity_col = "article_id"
    entity_table = "article"
    time_col = "timestamp"
    target_col = "transaction_count"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                tr.article_id,
                COUNT(tr.article_id) AS transaction_count
            FROM
                timestamp_df t
            LEFT JOIN
                transactions tr
            ON
                tr.t_dat > t.timestamp AND
                tr.t_dat <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                -- Only predict for articles sold in the last 3 months
                tr.article_id IN (
                    SELECT DISTINCT article_id
                    FROM transactions
                    WHERE t_dat BETWEEN t.timestamp - INTERVAL '3 months' AND t.timestamp
                )
            GROUP BY
                t.timestamp, tr.article_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"article_id": "article"},
            pkey_col=None,
            time_col="timestamp",
        )

class CustomerAvgPriceTask(EntityTask):
    r"""Predict the average price of items a customer will purchase in the
    next 7 days."""

    task_type = TaskType.REGRESSION
    entity_col = "customer_id"
    entity_table = "customer"
    time_col = "timestamp"
    target_col = "avg_price"
    timedelta = pd.Timedelta(days=7)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        transactions = db.table_dict["transactions"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                tr.customer_id,
                AVG(tr.price) AS avg_price
            FROM
                timestamp_df t
            LEFT JOIN
                transactions tr
            ON
                tr.t_dat > t.timestamp AND
                tr.t_dat <= t.timestamp + INTERVAL '{self.timedelta}'
            WHERE
                -- Only predict for customers who were active in the last 6 months
                tr.customer_id IN (
                    SELECT DISTINCT customer_id
                    FROM transactions
                    WHERE t_dat BETWEEN t.timestamp - INTERVAL '6 months' AND t.timestamp
                )
            GROUP BY
                t.timestamp, tr.customer_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={"customer_id": "customer"},
            pkey_col=None,
            time_col="timestamp",
        )