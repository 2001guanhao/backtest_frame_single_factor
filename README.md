# 量化回测框架 支持单因子分层 
 输出统计指标 可视化超额收益PnL累积IC 分层PnL
# 回测框架（Backtest Framework）

## 简介

本回测框架旨在为金融数据分析和量化交易策略的测试提供一个灵活、高效的平台。通过Python的面向对象编程和数据处理库，用户可以快速构建、测试和评估各种交易策略。

## 功能特性

- **数据灵活性**：支持多种数据输入格式，方便用户导入自己的数据集。
- **简洁的类设计**：使用`@dataclass`装饰器，简化类的定义和初始化，使代码更具可读性和维护性。
- **自动数据初始化**：通过`__post_init__`方法，在实例化对象后自动初始化数据，准备回测环境。
- **交易成本模拟**：支持设置印花税和券商佣金，真实模拟交易过程中的成本。
- **多策略支持**：可集成不同的交易策略，支持因子模型、权重优化等高级功能。
- **灵活的时间范围**：可自定义回测的起始和结束日期，灵活控制回测区间。
- **基准设定**：支持设置基准指数，例如沪深300，用于对比策略表现。

## 主要类和方法

### `Simultrading` 类

核心的模拟交易类，负责回测过程的初始化和执行。

#### 属性

- `start_date`：回测开始日期（字符串），格式为`'YYYY-MM-DD'`。
- `end_date`：回测结束日期（字符串），格式为`'YYYY-MM-DD'`。
- `stamp_duty`：印花税（浮点数），默认为`0.0`。可设置为`0.0005`（万分之五）。
- `brokerage_commission`：券商佣金（浮点数），默认为`0.0`。可设置为`0.0003`（万分之三）。
- `start_value`：初始资金（浮点数），默认为`1e+10`（一亿元）。
- `rebalance_df`：调仓数据（`pandas.DataFrame`）。要求双层索引，`level 0`为`'date'`，`level 1`为`'stock_code'`，列名包括`'factor'`、`'weight'`。
- `benchmark`：基准指数（字符串），默认为`'HS300'`。
- `close_df`：股票收盘价数据（`pandas.DataFrame`）。列名为股票代码（如`'000001.XSHG'`），索引为`'date'`。
- `strategy_name`：策略名称（字符串），默认为`'lstm'`。

#### 方法

- `__post_init__(self)`：数据类的初始化方法，在对象实例化后自动调用，负责初始化数据和准备回测。
- `initialize_data(self)`：数据初始化方法，执行以下操作：
  - 将`close_df`的索引转换为`datetime`类型。
  - 清理`rebalance_df`中的空值。
  - 获取回测日期范围，根据用户输入的`start_date`和`end_date`进行调整，确保在有效范围内。
- `prepare_backtest(self)`：准备回测环境的方法，具体实现根据项目需求进行扩展。

### `single_factor_test` 类

单因子测试类，用于评估单一因子在回测期间的表现。

#### 属性

- `num_layer` (`int`): 层数，默认为5。
- `end_date` (`str`, 可选): 回测结束日期，格式为`'YYYY-MM-DD'`。如果为`None`，则使用`rebalance_df`中的最后一个日期。
- `start_date` (`str`, 可选): 回测开始日期，格式为`'YYYY-MM-DD'`。如果为`None`，则使用`rebalance_df`中的第一个日期。
- `periods` (`tuple`): 回测周期，默认为`(1, 5, 10, 20)`，分别对应日频、周频、月频。
- `factor_df` (`pd.DataFrame`, 可选): 因子数据DataFrame。要求双索引，`level 0`为`'date'`，`level 1`为`'stock_code'`，且包含唯一的因子值列。
- `close_df` (`pd.DataFrame`, 可选): 股票收盘价数据。要求列名形式为`'000000.XSHG'`（与`rebalance_df`保持一致），索引为`'date'`。建议进行后向填充，以提高数据的完整性。
- `percent_ls` (`float`): 做多和做空的百分比，默认为0.1（即前10%做多，后10%做空）。
- `benchmark` (`str`): 基准指数，默认为`'HS300'`。
- `factor_name` (`str`, 可选): 因子名称，用于标识和展示。

#### 方法

- `__post_init__(self)`:
  - 将`factor_df`的列名设置为`'value'`。
  - 初始化统计结果DataFrame `stat_df`，包括以下指标：
    - `IC`: 信息系数
    - `ICIR`: 信息系数收益比
    - `L&S return`: 做多做空策略回报
    - `L return`: 做多策略回报
    - `market return`: 市场回报
    - `L&S Sharpe`: 做多做空策略夏普率
    - `L Sharpe`: 做多策略夏普率
    - `market Sharpe`: 市场夏普率
    - `L&S turnover`: 做多做空策略换手率
    - `L turnover`: 做多策略换手率
  - 调用`add_weight_ls`方法生成权重数据`df_factor_ls`。

- `resample_df_by_period(self, df: pd.DataFrame, period: int) -> pd.DataFrame`:
  - 按照给定的周期对DataFrame进行重采样。
  - 每隔`period`天选取一个日期，生成新的DataFrame。

- `period_test(self)`:
  - 对不同周期和策略（L&S、Long）进行回测。
  - 使用`joblib`的`Parallel`和`delayed`进行并行化处理。
  - 将回测结果填入`stat_df`中，并展示统计结果。

- `add_weight(self, df, i)`:
  - 根据因子值按日期分层，为第`i`层的股票赋予等权重，其他层权重为0。
  - 返回包含`'weight'`列的新DataFrame。

- `add_weight_ls(self, df, factor_column='value', top_percent=0.1, bottom_percent=0.1)`:
  - 按照因子值生成`'weight'`列。
  - 前`top_percent`百分比的股票赋予正等权重，后`bottom_percent`百分比的股票赋予负等权重，其余股票权重为0。
  - 确保总权重为0。
  - 返回更新后的DataFrame。

- `layer_test(self)`:
  - 对不同层次的组合进行回测，包括L&S策略。
  - 使用并行计算加快回测速度。
  - 绘制每个组合的收益曲线，并显示年化收益率。

- `show_performance(self)`:
  - 调用`layer_test`和`period_test`方法，展示回测的性能结果。

## 使用方法

1. **准备数据**：确保`rebalance_df`和`close_df`按上述格式准备且有效。

2. **安装依赖**：

   ```bash
   pip install -r requirements.txt
