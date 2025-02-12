# Biblioteca time-series-forecasting

# Documentação da Classe TimeSeriesForecasting

## Introdução
A classe `TimeSeriesForecasting` implementa uma solução para previsão de séries temporais utilizando o modelo **Prophet**, bem como uma abordagem baseada em **média móvel**. A classe está integrada ao **PySpark** para processamento distribuído, permitindo a aplicação dessas técnicas em grandes volumes de dados.

## Inicialização da Classe

```python
class TimeSeriesForecasting:
    def __init__(
        self,
        future_periods=15,
        holidays_prophet=None,
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

## Exemplo de Uso no PySpark (Sem Coluna de Classificação)

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

## Exemplo de Uso no PySpark (Cem Coluna de Classificação)

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from time_series_forecasting import TimeSeriesForecasting
import pandas as pd

# Criar DataFrame de exemplo com uma coluna de classificação
data = [("2024-01-01", "A", 100), ("2024-01-02", "A", 120), ("2024-01-03", "A", 130)]
columns = ["ds", "class", "y"]

df = spark.createDataFrame(data, columns)

columns_pyspark_class = df.columns

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


## Considerações Finais
Esta implementação permite previsões flexíveis utilizando Prophet ou Média Móvel, sendo eficiente para execução distribuída em PySpark.
O código é otimizado para grandes volumes de dados, garantindo escalabilidade.

