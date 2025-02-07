from prophet import Prophet
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from datetime import timedelta
import pandas as pd
import numpy as np


class TimeSeriesForecasting:
    def __init__(
        self,
        future_periods=15,
        holidays_prophet=None,
        seasonalities_prophet=None,
        columns_pyspark=None,
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
            model = Prophet(holidays=self.holidays_prophet)
        else:
            model = Prophet()

        # Adiciona as sazonalidades configuradas
        if self.seasonalities_prophet:
            for seasonality in self.seasonalities_prophet:
                model.add_seasonality(
                    name=seasonality["name"],
                    period=seasonality["period"],
                    fourier_order=seasonality["fourier_order"],
                )

        model.fit(df_partition)

        future = model.make_future_dataframe(periods=self.future_periods)
        forecast = model.predict(future)

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
            return np.mean(last_n[-n_values:]) if last_n else np.nan

        def std_last_n_numbers(values, index):
            # Obter os últimos N valores diferentes de zero antes do índice atual
            last_n = [v for v in values[:index] if v != 0 and not pd.isnull(v)]
            return round(np.std(last_n[-n_values:]), 0) if last_n else np.nan

        n_values = 30

        df_partition = df_partition.set_index("ds")
        mean_column_values = []
        std_column_values = []

        # Adicionando 14 novos registros com a quantidade preenchida como zero
        last_date_hour = df_partition.index.max()  # Obtém a última data e hora
        new_date_hour = [
            last_date_hour + timedelta(hours=i + 1) for i in range(self.future_periods)
        ]
        new_load = pd.DataFrame({"ds": new_date_hour, "y": [0] * self.future_periods})
        new_load.set_index("ds", inplace=True)

        df_complete = pd.concat([df_partition, new_load], axis=0)

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
        df_complete["yhat_upper"] = df_complete["yhat"] + 2 * df["std"]

        df_complete = df_complete.drop(["std"], axis=1)
        df_complete = df_complete.clip(lower=0, upper=None)

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
