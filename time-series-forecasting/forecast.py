from prophet import Prophet
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType


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
        model = Prophet(holidays=self.holidays_prophet)

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
