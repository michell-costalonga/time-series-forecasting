# Documentação da Classe TimeSeriesForecasting

## Introdução
A classe `TimeSeriesForecasting` implementa uma solução para previsão de séries temporais utilizando o modelo **Prophet**, bem como uma abordagem baseada em **média móvel**. A classe está integrada ao **PySpark** para processamento distribuído, permitindo a aplicação dessas técnicas em grandes volumes de dados.

## Inicialização da Classe

```python
class TimeSeriesForecasting:
    def __init__(
        self,
        future_periods=15,
        holidays_prophet=pd.DataFrame(),
        seasonalities_prophet=None,
        seasonality_mode_prophet="additive",
        columns_pyspark=None,
        frequency="daily",
    )
```

### Parâmetros
- `future_periods` *(int)*: Quantidade de períodos futuros a serem previstos.
- `holidays_prophet` *(DataFrame Pandas, opcional)*: Conjunto de feriados para auxiliar o modelo Prophet.
- `seasonalities_prophet` *(lista de dicionários, opcional)*: Configuração de sazonalidades personalizadas.
- `seasonality_mode_prophet` *(str, "additive" ou "multiplicative")*: Define o modo da sazonalidade.
- `columns_pyspark` *(lista, opcional)*: Colunas para serem consideradas na segmentação dos dados no PySpark.
- `frequency` *(str)*: Frequência da série temporal (*hourly, daily, weekly, monthly, yearly*).

## Métodos

### `fit_prophet_pandas(df_partition)`

Ajusta o modelo Prophet a um DataFrame Pandas.

**Parâmetros:**
- `df_partition` *(DataFrame Pandas)*: DataFrame contendo as colunas `ds` (data) e `y` (valor).

**Retorno:**
- DataFrame Pandas com previsões (`ds`, `yhat`, `yhat_lower`, `yhat_upper`).

---

### `get_prophet_udf()`

Retorna uma **UDF** (User Defined Function) do PySpark para aplicação do modelo Prophet em grupos de dados.

**Retorno:**
- Função UDF para uso com PySpark.

---

### `fill_with_mean_and_std(df_partition)`

Realiza previsão baseada em **média móvel** e desvio padrão, preenchendo valores futuros.

**Parâmetros:**
- `df_partition` *(DataFrame Pandas)*: DataFrame contendo `ds` e `y`.

**Retorno:**
- DataFrame Pandas com previsões (`ds`, `yhat`, `yhat_lower`, `yhat_upper`).

---

### `get_average_udf()`

Retorna uma **UDF** para aplicação do método de **média móvel** no PySpark.

**Retorno:**
- Função UDF para uso com PySpark.

---

## Instalação da Biblioteca

Para instalar a biblioteca diretamente do **GitHub** você pode usar o comando **pip** dentro do notebook.

```python
pip install git+https://github.com/michell-costalonga/time-series-forecasting
```

ou

```python
from sys import executable
from subprocess import check_call, DEVNULL
from pkg_resources import working_set

time_series_forecasting_lib = "git+https://github.com/michell-costalonga/time-series-forecasting"
required = {time_series_forecasting_lib}
installed = {pkg.key for pkg in working_set}
missing = required - installed
if missing:
    python = executable
    check_call([python, "-m", "pip", "install", *missing], stdout=DEVNULL)
```

## Exemplo de Uso no PySpark (Sem Coluna de Classificação) - Prophet

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from time_series_forecasting import TimeSeriesForecasting
import pandas as pd

# Criar DataFrame de exemplo sem uma coluna de classificação
data = [("2024-01-01", 100), ("2024-01-02", 120), ("2024-01-03", 130)]
columns = ["ds", "y"]

df = spark.createDataFrame(data, columns)

columns_pyspark_class = df.columns

modeler = TimeSeriesForecasting(
    future_periods=15, 
    columns_pyspark=columns_pyspark,
)

# Obter UDF para previsão
prophet_udf = modeler.get_prophet_udf()

# Aplicar previsão usando PySpark
df_forecast = df.groupBy().apply(prophet_udf)
display(df_forecast)
```

## Exemplo de Uso no PySpark (Cem Coluna de Classificação) - Prophet

```python
from pyspark.sql.functions import col
from time_series_forecasting import TimeSeriesForecasting
import pandas as pd

# Criar DataFrame de exemplo com uma coluna de classificação
data = [("2024-01-01", "A", 100), ("2024-01-02", "A", 120), ("2024-01-03", "A", 130)]
columns = ["ds", "class", "y"]

df = spark.createDataFrame(data, columns)

columns_pyspark = df.columns

modeler = TimeSeriesForecasting(
    future_periods=15, 
    columns_pyspark=columns_pyspark,
)

# Obter UDF para previsão
prophet_udf = modeler.get_prophet_udf()

# Aplicar previsão usando PySpark
df_forecast = df.groupBy("class").apply(prophet_udf)
display(df_forecast)
```

## Exemplo de Uso no PySpark (Com Coluna de Classificação) - Média Móvel

```python
from pyspark.sql.functions import col
from time_series_forecasting import TimeSeriesForecasting
import pandas as pd

# Criar DataFrame de exemplo com uma coluna de classificação
data = [("2024-01-01", "A", 100), ("2024-01-02", "A", 120), ("2024-01-03", "A", 130)]
columns = ["ds", "class", "y"]

df = spark.createDataFrame(data, columns)

columns_pyspark = df.columns

modeler = TimeSeriesForecasting(
    future_periods=15, 
    columns_pyspark=columns_pyspark,
)

# Obter UDF para previsão
avg_udf = modeler.get_average_udf()

# Aplicar previsão usando PySpark
df_forecast = df.groupBy("class").apply(avg_udf)
display(df_forecast)
```

# Funções complementares

## Exemplo de uso rename_columns
A função auxilia na transformação do DataFrame para o formato esperado pela classe `TimeSeriesForecasting`.

```python
from pyspark.sql.functions import col
from time_series_forecasting import rename_columns
import pandas as pd

# Criar DataFrame de exemplo com uma coluna de classificação
data = [
    ("2024-01-01", "A", 100), ("2024-01-02", "A", 120), ("2024-01-03", "A", 130),
]

columns = ["created_date", "class", "quantity"]
df_raw = spark.createDataFrame(data, columns)

# Usando a função para renomear as colunas
renamed_df = rename_columns(
    df_pyspark_agg=df_raw,
    target_column_name='quantity',
    date_column_name='created_date'
)

display(renamed_df)
```

## Exemplo de uso filter_dataframe
A função `filter_dataframe` pode ser usada para separar o DataFrame principal em dois outros DataFrames de acordo com o volume de dados.
A entrada da função é o DataFrame contendo todos os dados e saída da função retorna o `model_df`, que é o DataFrame indicado para o uso com o modelo de previsão da classe `TimeSeriesForecasting`, e o `avg_df` que é o DataFrame indicado para aplicar o cálculo com a média móvel.

```python
from pyspark.sql.functions import col
from time_series_forecasting import filter_dataframe
import pandas as pd

# Criar DataFrame de exemplo com uma coluna de classificação
data = [
    ("2024-01-01", "A", 100), ("2024-01-02", "A", 120), ("2024-01-03", "A", 130),
    ("2024-01-04", "A", 110), ("2024-01-05", "A", 115), ("2024-01-06", "A", 140),
    ("2024-01-07", "A", 125), ("2024-01-08", "A", 135), ("2024-01-09", "A", 150),
    ("2024-01-10", "A", 160),  

    ("2024-01-01", "B", 90), ("2024-01-02", "B", 95), ("2024-01-03", "B", 100),
    ("2024-01-04", "B", 110), ("2024-01-05", "B", 105), ("2024-01-06", "B", 115),
    ("2024-01-07", "B", 120), ("2024-01-08", "B", 125),  ("2024-01-09", "B", 130),

    ("2024-01-01", "C", 80), ("2024-01-02", "C", 85), ("2024-01-03", "C", 90),
]

columns = ["ds", "class", "y"]
df = spark.createDataFrame(data, columns)

# Usando a função para renomear as colunas
model_df, avg_df = filter_dataframe(df)

print('DataFrame que pode ser usado com o modelo de forecast:')
model_df.show()

print('DataFrame que pode ser usado com o cálculo da média móvel:')
avg_df.show()
```

## Exemplo Completo
```python
from pyspark.sql.functions import col
from time_series_forecasting import TimeSeriesForecasting, rename_columns, filter_dataframe
import pandas as pd

# Criar DataFrame de exemplo com uma coluna de classificação
data = [
    ("2024-01-01", "A", 100), ("2024-01-02", "A", 120), ("2024-01-03", "A", 130),
    ("2024-01-04", "A", 110), ("2024-01-05", "A", 115), ("2024-01-06", "A", 140),
    ("2024-01-07", "A", 125), ("2024-01-08", "A", 135), ("2024-01-09", "A", 150),
    ("2024-01-10", "A", 160),  

    ("2024-01-01", "B", 90), ("2024-01-02", "B", 95), ("2024-01-03", "B", 100),
    ("2024-01-04", "B", 110), ("2024-01-05", "B", 105), ("2024-01-06", "B", 115),
    ("2024-01-07", "B", 120), ("2024-01-08", "B", 125),  ("2024-01-09", "B", 130),

    ("2024-01-01", "C", 80), ("2024-01-02", "C", 85), ("2024-01-03", "C", 90),
]

columns = ["created_date", "class", "quantity"]
df_raw = spark.createDataFrame(data, columns)

# Usando a função para renomear as colunas
renamed_df = rename_columns(
    df_pyspark_agg=df_raw,
    target_column_name='quantity',
    date_column_name='created_date'
)

# Coletando o nome das colunas do DataFrame principal
columns_pyspark = renamed_df.columns

# Usando a função para renomear as colunas
model_df, avg_df = filter_dataframe(renamed_df)

# Iniciando classe
modeler = TimeSeriesForecasting(
    future_periods=15, 
    columns_pyspark=columns_pyspark,
)

# Obter UDF para previsão - Prophet
model_udf = modeler.get_prophet_udf()

# Obter UDF para previsão - Média Móvel
avg_udf = modeler.get_average_udf()

# Aplicar previsão usando PySpark
df_forecast_model = model_df.groupBy("class").apply(model_udf)
df_forecast_avg = avg_df.groupBy("class").apply(avg_udf)

df_forecast = df_forecast_model.union(df_forecast_avg)
display(df_forecast)

```

## Considerações Finais
Esta implementação permite previsões flexíveis utilizando Prophet ou Média Móvel, sendo eficiente para execução distribuída em PySpark.
O código é otimizado para grandes volumes de dados, garantindo escalabilidade.

