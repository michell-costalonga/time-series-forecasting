from prophet import Prophet
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np


class TimeSeriesForecasting:
    def __init__(
        self,
        future_periods=15,
        holidays_prophet=None,
        seasonalities_prophet=None,
        seasonality_mode_prophet="additive",
        columns_pyspark=None,
        frequency="daily",
    ):
        """
        Inicializa o modelador Prophet com configurações flexíveis.

        :param holidays_prophet: Pandas DataFrame de feriados para o Prophet.
        :param future_periods: Número de períodos futuros para previsão.
        :param seasonalities_prophet: Lista de dicionários para sazonalidades personalizadas para o Prophet.
        """
        self.future_periods = future_periods
        self.holidays_prophet = holidays_prophet
        self.columns_pyspark = columns_pyspark

        # Sazonalidades personalizadas (se nenhuma for passada, usa valores padrão)
        self.seasonalities_prophet = seasonalities_prophet
        self.seasonality_mode_prophet = seasonality_mode_prophet

        self.frequency = frequency
        self.dict_frequency = {
            "hourly": "H",
            "daily": "D",
            "weekly": "W",
            "monthly": "MS",
            "yearly": "YS",
        }

    def fit_prophet_pandas(self, df_partition):
        """
        Ajusta o modelo Prophet a um DataFrame Pandas.

        :param df_partition: DataFrame Pandas com as colunas 'ds' e 'y'.
        :return: DataFrame com previsões.
        """
        df_partition = df_partition.fillna(0)
        df_partition = df_partition.sort_values(by="ds", ascending=True)

        # Inicializa o modelo Prophet com ou sem feriados
        if not self.holidays_prophet.empty:
            model = Prophet(
                seasonality_mode=self.seasonality_mode_prophet,
                holidays=self.holidays_prophet,
            )
        else:
            model = Prophet(seasonality_mode=self.seasonality_mode_prophet)

        # Adiciona as sazonalidades configuradas
        if self.seasonalities_prophet:
            for seasonality in self.seasonalities_prophet:
                model.add_seasonality(
                    name=seasonality["name"],
                    period=seasonality["period"],
                    fourier_order=seasonality["fourier_order"],
                )

        model.fit(df_partition)

        future = model.make_future_dataframe(
            periods=self.future_periods, freq=self.dict_frequency[self.frequency]
        )
        forecast = model.predict(future)

        if self.frequency == "hourly":
            forecast["ds"] = forecast["ds"].astype("datetime64[ns]").dt.floor("H")
        else:
            forecast["ds"] = forecast["ds"].astype("datetime64[ns]").dt.floor("D")

        cols_to_clip = ["yhat", "yhat_lower", "yhat_upper"]
        forecast[cols_to_clip] = forecast[cols_to_clip].clip(lower=0, upper=None)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def get_prophet_udf(self):
        """
        Retorna uma UDF configurada para aplicar o Prophet em grupos.

        :return: Função UDF que pode ser usada com PySpark.
        """
        pandas_schema = "ds timestamp, yhat double, yhat_lower double, yhat_upper double, model string"

        if self.columns_pyspark:
            columns_name = [
                col for col in self.columns_pyspark if col not in ["ds", "y"]
            ]
            for column_name in columns_name:
                pandas_schema += f", {column_name} string"

        @pandas_udf(
            pandas_schema,
            PandasUDFType.GROUPED_MAP,
        )
        def fit_prophet_udf(df):
            # Extrair valores de agrupamento (assumindo que as colunas estão no DataFrame)
            columns = [col for col in df.columns if col not in ["ds", "y"]]
            list_values = []
            if columns:
                for column in columns:
                    list_values.append(df[column].iloc[0])

            # Ajustar o modelo Prophet
            forecast = self.fit_prophet_pandas(df[["ds", "y"]])

            # Inserindo o nome do algoritmo usado
            forecast["model"] = "Prophet"

            # Adicionar colunas de agrupamento ao resultado
            if columns:
                for i, column in enumerate(columns):
                    forecast[column] = list_values[i]

            return forecast

        return fit_prophet_udf

    def fill_with_mean_and_std(self, df_partition):
        def avg_last_n_numbers(values, index):
            # Obter os últimos N valores diferentes de zero antes do índice atual
            last_n = [v for v in values[:index] if v != 0 and not pd.isnull(v)]
            return round(np.mean(last_n[-n_values:]), 0) if last_n else np.nan

        def std_last_n_numbers(values, index):
            # Obter os últimos N valores diferentes de zero antes do índice atual
            last_n = [v for v in values[:index] if v != 0 and not pd.isnull(v)]
            return round(np.std(last_n[-n_values:]), 0) if last_n else np.nan

        n_values = 30

        df_partition = df_partition.set_index("ds").asfreq(
            self.dict_frequency[self.frequency]
        )
        mean_column_values = []
        std_column_values = []

        # Adicionando 14 novos registros com a quantidade preenchida como zero
        date_hour_df = df_partition.index.max()  # Obtém a última data e hora
        current_day = datetime.strptime(
            (datetime.today() - timedelta(hours=3)).strftime("%Y-%m-%d 00:00"),
            "%Y-%m-%d 00:00",
        )

        if date_hour_df < current_day:
            last_date_hour = current_day
        else:
            last_date_hour = date_hour_df

        if self.frequency == "hourly":
            new_date_hour = [
                last_date_hour + timedelta(hours=i) for i in range(self.future_periods)
            ]
        elif self.frequency == "daily":
            new_date_hour = [
                last_date_hour + timedelta(days=i) for i in range(self.future_periods)
            ]
        elif self.frequency == "weekly":
            new_date_hour = [
                last_date_hour + relativedelta(weeks=i)
                for i in range(self.future_periods)
            ]
        elif self.frequency == "monthly":
            new_date_hour = [
                last_date_hour + relativedelta(months=i)
                for i in range(self.future_periods)
            ]
        elif self.frequency == "yearly":
            new_date_hour = [
                last_date_hour + relativedelta(years=i)
                for i in range(self.future_periods)
            ]
        else:
            print("Frequência não válida.")

        new_load = pd.DataFrame({"ds": new_date_hour, "y": [0] * self.future_periods})
        new_load.set_index("ds", inplace=True)

        df_complete = pd.concat([df_partition, new_load], axis=0)
        df_complete["ds"] = pd.to_datetime(df_complete["ds"])
        df_complete = df_complete.asfreq(
            self.dict_frequency[self.frequency], fill_value=0
        )

        values = df_complete["y"].tolist()

        for i in range(len(values)):
            avg_values = avg_last_n_numbers(values, index=i)
            std_values = std_last_n_numbers(values, index=i)
            mean_column_values.append(
                int(avg_values if not pd.isnull(avg_values) else 0)
            )
            std_column_values.append(std_values if not pd.isnull(std_values) else 0)

        df_complete["yhat"] = mean_column_values
        df_complete["std"] = std_column_values
        df_complete["yhat_lower"] = df_complete["yhat"] - 2 * df_complete["std"]
        df_complete["yhat_upper"] = df_complete["yhat"] + 2 * df_complete["std"]

        df_complete = df_complete.drop(["std"], axis=1)
        df_complete = df_complete.clip(lower=0, upper=None)

        df_complete = df_complete.reset_index()
        df_complete["ds"] = df_complete["ds"].astype("datetime64[ns]")

        return df_complete[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def get_average_udf(self):
        """
        Retorna uma UDF configurada para aplicar a média em grupos.

        :return: Função UDF que pode ser usada com um PySpark DataFrame.
        """
        pandas_schema = "ds timestamp, yhat double, yhat_lower double, yhat_upper double, model string"

        if self.columns_pyspark:
            columns_name = [
                col for col in self.columns_pyspark if col not in ["ds", "y"]
            ]
            for column_name in columns_name:
                pandas_schema += f", {column_name} string"

        @pandas_udf(
            pandas_schema,
            PandasUDFType.GROUPED_MAP,
        )
        def fill_moving_average_udf(df):
            # Extrair valores de agrupamento (assumindo que as colunas estão no DataFrame)
            columns = [col for col in df.columns if col not in ["ds", "y"]]
            list_values = []
            if columns:
                for column in columns:
                    list_values.append(df[column].iloc[0])

            # Ajustar o modelo Prophet
            forecast = self.fill_with_mean_and_std(df[["ds", "y"]])

            # Inserindo o nome do algoritmo usado
            forecast["model"] = "Moving Average"

            # Adicionar colunas de agrupamento ao resultado
            if columns:
                for i, column in enumerate(columns):
                    forecast[column] = list_values[i]

            return forecast

        return fill_moving_average_udf
