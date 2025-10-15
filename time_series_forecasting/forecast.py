from prophet import Prophet
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, fbeta_score


class TimeSeriesForecasting:
    def __init__(
        self,
        future_periods=15,
        holidays_prophet=pd.DataFrame(),
        seasonalities_prophet=None,
        seasonality_mode_prophet="additive",
        columns_pyspark=None,
        frequency="daily",
        identifying_peaks=False,
        quantile_value=0.975,
        lookback_value=5,
        fill_value_metric="median"
    ):
        """
        Inicializa o modelador Prophet com configurações flexíveis.

        :param holidays_prophet: Pandas DataFrame de feriados para o Prophet.
        :param future_periods: Número de períodos futuros para previsão.
        :param seasonalities_prophet: Lista de dicionários para sazonalidades personalizadas para o Prophet.
        :param seasonalities_mode_prophet: String definindo o modo de sazonalidade: 'additive' ou 'multiplicative'.
        :param columns_pyspark: Lista de strings contendo o nome das colunas.
        :param frequency: String indicando a frequência da série temporal: 'monthly', 'daily', 'hourly'.
        :param quantile_value: Float que indica o valor limite do quantil para identificar o pico.
        :param fill_value_metric: Métrica que será usada para preencher os valores de picos faltantes. Pode ser 'mean' ou 'median'.
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

        self.identifying_peaks = identifying_peaks
        self.lookback_value = lookback_value
        self.quantile_value = quantile_value
        self.fill_value_metric = fill_value_metric

    def fit_prophet_pandas(self, df_partition):
        """
        Ajusta o modelo Prophet a um DataFrame Pandas.

        :param df_partition: DataFrame Pandas com as colunas 'ds' e 'y'.
        :return: DataFrame com previsões.
        """
        df = df_partition.fillna(0)
        df = df.drop_duplicates(subset="ds", keep="last")
        df = df.sort_values(by="ds", ascending=True)

        series = pd.Series(df["y"].values, index=pd.DatetimeIndex(df["ds"])).asfreq(self.dict_frequency[self.frequency])
        series = series.interpolate()

        threshold = series.quantile(self.quantile_value)
        is_spike = (series > threshold).astype(int)

        df_partition = pd.DataFrame({
            "ds": series.index,
            "y": series.values
        })

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

        forecast_baseline = forecast.set_index("ds")["yhat_upper"].values #.iloc[-self.future_periods:]
        future_index = forecast["ds"]

        if self.identifying_peaks == True:

            # Principais feriados do Brasil
            actual_year = (datetime.now() - timedelta(hours=3)).year
            next_year = actual_year + 1
            year_before = actual_year - 1
            holidays_br = holidays.Brazil(years=[year_before, actual_year, next_year])

            # Classificador de picos
            df_features = pd.DataFrame({
                "y": series,
                "is_spike": is_spike,
                "hour": series.index.hour,
                "day": series.index.day,
                "dayofweek": series.index.dayofweek,
                "is_holiday": pd.Series(series.index.date).isin(holidays_br).values
            }).dropna()

            X = df_features[[column for column in df_features.columns if column not in ["y","is_spike"]]]
            y_spike = df_features["is_spike"]

            # Modelo para identificar picos

            param_grid = {
                "n_estimators": [100, 300, 500],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            }

            X_train, X_test, y_train, y_test = train_test_split(X, y_spike, test_size=0.2, shuffle=False)

            tscv = TimeSeriesSplit(n_splits=5)
            fbeta_scorer = make_scorer(fbeta_score, beta=1.5)

            rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=tscv,
                scoring=fbeta_scorer,
                verbose=-1,
                n_jobs=-1
            )

            grid_search.fit(X, y_spike)

            clf = grid_search.best_estimator_

            # Previsão dos picos
            y_pred = clf.predict(X_test)

            # Combinar previsões
            def get_avg_spike_size(ts_index, series, is_spike, lookback=2):
                """
                Calcula a média dos tamanhos de pico para a mesma hora e mesmo dia da semana,
                olhando ocorrências anteriores (o período é definido pela variável 'lookback').
                """
                dow = ts_index.dayofweek
                hour = ts_index.hour

                mask = (series.index.dayofweek == dow) & (series.index.hour == hour) & (is_spike == 1)
                history = series[mask]

                if len(history) >= lookback:
                    return history.iloc[-lookback:].mean()
                elif len(history) > 0:
                    return history.mean()
                else:
                    # fallback para média global de picos
                    if self.fill_value_metric == "median":
                        return series[is_spike == 1].median()
                    else:
                        return series[is_spike == 1].mean()

            future_df = pd.DataFrame({ 
                "hour": future_index.dt.hour, 
                "day": future_index.dt.day, 
                "dayofweek": future_index.dt.dayofweek, 
                "is_holiday": future_index.dt.date.isin(holidays_br)
            })

            spike_pred_future = clf.predict(future_df)

            # Calcular forecast final ponto a ponto
            forecast_final = []
            for ds, base, spike_flag in zip(future_index, forecast_baseline, spike_pred_future):
                if spike_flag == 1:
                    spike_size = get_avg_spike_size(ds, series, is_spike, lookback=self.lookback_value)
                    forecast_final.append(spike_size*1.1)
                else:
                    forecast_final.append(base)

            forecast_final = np.array(forecast_final)
            forecast["yhat_upper"] = forecast_final

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

        mean_column_values = []
        std_column_values = []

        # Adicionando novos registros com a quantidade preenchida como zero
        # Obtém a última data e hora
        if self.frequency == "hourly":
            try:
                date_hour_df = datetime.strptime(
                    str(df_partition["ds"].max()), "%Y-%m-%d %H:%M:%S"
                )
            except:
                date_hour_df = datetime.strptime(
                    str(df_partition["ds"].max()), "%Y-%m-%d %H:%M"
                )
        else:
            try:
                date_hour_df = datetime.strptime(
                    str(df_partition["ds"].max()), "%Y-%m-%d %H:%M:%S"
                )
            except:
                date_hour_df = datetime.strptime(
                    str(df_partition["ds"].max()), "%Y-%m-%d"
                )
        current_day = datetime.strptime(
            (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d 00:00"),
            "%Y-%m-%d 00:00",
        )

        if date_hour_df < current_day:
            last_date_hour = current_day
        else:
            last_date_hour = date_hour_df

        if self.frequency == "hourly":
            new_date_hour = [
                last_date_hour + timedelta(hours=i + 1)
                for i in range(self.future_periods)
            ]
        elif self.frequency == "daily":
            new_date_hour = [
                last_date_hour + timedelta(days=i + 1)
                for i in range(self.future_periods)
            ]
        elif self.frequency == "weekly":
            new_date_hour = [
                last_date_hour + relativedelta(weeks=i + 1)
                for i in range(self.future_periods)
            ]
        elif self.frequency == "monthly":
            new_date_hour = [
                last_date_hour + relativedelta(months=i + 1)
                for i in range(self.future_periods)
            ]
        elif self.frequency == "yearly":
            new_date_hour = [
                last_date_hour + relativedelta(years=i + 1)
                for i in range(self.future_periods)
            ]
        else:
            print("Frequência não válida.")

        new_load = pd.DataFrame({"ds": new_date_hour, "y": [0] * self.future_periods})

        df_complete = pd.concat([df_partition, new_load], axis=0)
        df_complete["ds"] = pd.to_datetime(df_complete["ds"])
        df_complete = df_complete.set_index("ds").asfreq(
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
    