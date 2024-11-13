# 量化回测框架、投资组合优化器以及单因子测试模块 

## 目录

1. 简介
2. 模块详解
   - 回测框架
   - 投资组合优化器
   - 单因子测试
3. 数据流与执行流程
4. 指标与结果分析
5. 使用方法与示例
6. 未来改进方向
7. 总结

---

## 简介

量化投资回测框架，包含**回测框架**、**投资组合优化器**和**单因子测试**三个模块。通过该框架，用户仅需提供因子数据，即可完成完整的回测流程，并对策略进行深入分析。框架支持并行计算，能够在大数据量的情况下快速运行。框架粗糙，还有太多可改进之处，非常欢迎大家一起来体验使用，参与改进，量化学习不应该闭门造车、讳莫如深，愿和CS、AI领域一样有开源的精神。

---

## 模块详解

### 回测框架

回测框架作为核心模块，整合了投资组合优化器，同时也是单因子测试模块的基础，封装了一套完整的策略回测流程。

**主要功能：**

- **策略回测**：根据优化后的权重数据，执行交易，计算策略绩效。
- **集成优化**：内置投资组合优化器，支持直接输入因子数据进行优化和回测。
- **资金管理**：处理资金分配、持仓管理、交易成本计算等。
- **止盈止损策略**：支持多种止盈止损方法，如固定比例、移动均线、ATR 等。
- **结果展示**：提供绘制净值曲线、收益分布等功能。
- **数据保存**：支持将回测结果、交易记录等保存为文件，便于后续分析。

**主要参数：**

```python
Simultrading(
    start_date,                  # 回测开始日期，格式 'YYYY-MM-DD'
    end_date,                    # 回测结束日期，格式 'YYYY-MM-DD'
    stamp_duty=0.0003,           # 印花税费率
    brokerage_commission=0.0005, # 券商佣金费率
    start_value=1e+6,            # 初始资金
    rebalance_df=None,           # 调仓数据，包含权重
    close_df=None,               # 收盘价数据
    strategy_name='',            # 策略名称
    benchmark=None,              # 基准指数代码或数据
    optimization={},             # 优化参数配置
    TP_SL={}                     # 止盈止损参数配置
)
```

**参数含义及使用方式：**

- **`start_date` / `end_date`**：指定回测的时间范围。
- **`stamp_duty` / `brokerage_commission`**：设置交易成本，印花税卖出时收取，佣金双向收取，如有其他税费也可通过设定这两个参数模拟。
- **`start_value`**：初始资金量。
- **`rebalance_df`**：最重要的df，包含调仓日期和对应股票权重的数据。格式为双索引('date','stock_code')，列名为'factor','weight'(若未提供weight列，将根据输入的`optimization`参数（包括优化目标选股数量等）进行优化生成。
- **`close_df`**：股票的收盘价数据，索引为'date'，列名为股票代码。
- **`strategy_name`**：策略名称，便于结果命名和识别。
- **`benchmark`**：基准指数（如`'HS300'`），用于计算超额收益或优化最小跟踪误差等。
- **`optimization`**：优化器参数，详见投资组合优化器。
- **`TP_SL`**：止盈止损策略配置。

**运行速度：**

- **三年数据量**：M1Pro 约 12 秒完成回测（日频）若需组合优化则约 19 秒，具体时间取决于数据量和计算机性能。

### 投资组合优化器

该模块用于根据因子数据和市场数据，对投资组合进行优化，计算每只股票的持仓权重。

**主要功能：**

- **股票选择**：基于因子值，选择符合条件的股票构建投资组合。
- **目标优化**：支持多种优化目标，如最大夏普比率、最小方差、风险平价等。
- **权重计算**：使用优化算法计算最优权重。
- **自动集成**：可与回测框架集成，仅需提供因子数据即可完成优化和回测。

**主要参数：**

```python
PortfolioOptimizer(
    close_df,                   # 收盘价数据 同上
    rebalance_df,               # 因子数据 有且仅有'factor'列 其余同上
    benchmark_close=None,       # 基准指数收盘价数据 单索引'date' 单列
    risk_aversion=1.0           # 风险厌恶系数
)
```

- **`close_df`**：股票收盘价数据，索引为日期，列名为股票代码。
- **`rebalance_df`**：调仓因子数据，索引为日期和股票代码，包含因子值。
- **`benchmark_close`**：基准指数收盘价数据，可用于相对指标的优化。
- **`risk_aversion`**：风险厌恶系数，控制收益与风险的权衡。

**优化方法：**

```python
optimize_weight(
    target='EqualWeight',       # 优化目标
    weight_bounds=(0, 1),       # 单支股票权重上下限
    regularization=None,        # 正则化参数 避免优化结果过于极端
    count=250                   # 估计均值协方差所用的历史数据长度
)
```

- **`target`**：优化目标，包括：
  - `'EqualWeight'`：等权重
  - `'MinVariance'`：最小方差
  - `'MaxSharpeRatio'`：最大夏普比率
  - `'RiskParity'`：风险平价
- **`weight_bounds`**：权重的上下限，默认在 `[0, 1]` 之间。
- **`regularization`**：正则化参数。
- **`count`**：计算协方差矩阵的历史数据长度。

**运行速度：**

- **三年数据量**：约 7 秒完成优化。

### 单因子测试

用于评估因子的有效性，包括计算因子 IC、分层收益、绘制图表等。

**主要功能：**

- **因子检验**：计算因子与未来收益的相关性（IC）。
- **分层回测**：将股票按因子值分层，测试不同层级的收益表现。
- **多周期分析**：支持多种调仓周期（如每日、每周、每月）。
- **多线程并行**：利用多线程技术，加速计算过程。
- **多空、多头策略**：支持多空组合、仅多头组合的收益分析。

**主要参数：**

```python
FactorTester(
    factor_df,                 # 因子数据
    close_df                   # 收盘价数据
)
```

- **`factor_df`**：因子数据，索引为日期和股票代码。
- **`close_df`**：股票收盘价数据。

**测试方法：**

```python
single_factor = single_factor_test(
    num_layer = 5, #分层数量
    end_date = None, #测试结束时间
    start_date = None, #测试开始时间
    periods= (1,5,10,20), # 调仓周期
    factor_df = pd.DataFrame(df_lstm.iloc[:,0]) , #因子数据dataframe 要求双索引 level0是date level1是stock_code 有唯一的因子值列
    close_df = df_close  ,# 股票收盘价数据 要求列名形式为'000000.XSHG'（与rebalance_df保持一致即可） index为‘date’ 建议后向填充一下 不填充的话更真实
    percent_ls = 0.1, #多空测试取因子值在前后10%百分比的股票
    factor_name= 'LSTM') #测试因子名称
single_factor.show_performance()
```

- **`num_layers`**：将股票按因子值等分为几层。
- **`periods`**：指定调仓周期。

**运行速度：**

- **三年数据量**：分五层 四个调仓周期 启用多线程约 33 秒完成测试。

---

## 数据流与执行流程

### 数据准备

1. **收盘价数据 `close_df`**：获取股票的历史收盘价数据，索引为日期，列名为股票代码。
2. **因子数据 `rebalance_df`**：生成或获取股票在调仓日的因子值，索引为日期和股票代码。

### 回测流程

1. **初始化回测框架**：提供必要的数据和参数。如仅提供因子数据，框架将自动调用优化器生成权重。
2. **每日迭代**：
   - **调仓日**：
     - 计算目标持仓权重。
     - 执行交易，更新持仓和现金。
   - **非调仓日**：
     - 根据市场价格更新持仓市值。
     - 执行止盈止损策略。
3. **结果输出**：
   - 生成每日净值、累计收益率等指标。
   - 提供绘图和数据保存功能。

### 单因子测试流程

1. **数据预处理**：对因子数据进行去极值、标准化等处理。
2. **分层测试**：
   - 将股票按因子值排序，等分为指定层数。
   - 根据调仓周期，计算各层级的平均收益。
3. **结果分析**：
   - 计算因子 IC、ICIR 等指标。
   - 绘制多空组合、各层级累计收益曲线等图表。

---

## 指标与结果分析

**主要指标：**

- **累计收益率**：策略在回测期间的总收益。
- **年化收益率**：将累计收益率换算成年化形式。
- **最大回撤**：历史最大净值回撤比例。
- **夏普比率**：风险调整后的收益指标。
- **信息系数（IC）**：因子值与未来收益率的相关性。
- **ICIR**：IC 的信息比率，衡量因子稳定性。

**图表描述：**

- **净值曲线**：展示策略净值随时间的变化。
- **回撤曲线**：显示策略的历史回撤情况。
- **分层收益曲线**：比较不同层级的累计收益。
- **IC 时序图**：展示因子 IC 的时间变化。

---

## 使用方法与示例

### 数据准备

**收盘价数据 `close_df`：**

```python
import pandas as pd

close_df = pd.read_csv('close_prices.csv', index_col='date', parse_dates=True)
```

**因子数据 `rebalance_df`：**

```python
rebalance_df = pd.read_csv('factor_data.csv', index_col=['date', 'stock_code'])
```

### 回测执行

```python
from backtesting_framework import Simultrading

# 初始化回测框架
backtest = Simultrading(
    start_date='2020-01-01',
    end_date='2022-12-31',
    close_df=close_df,
    rebalance_df=rebalance_df,
    optimization={
        'target': 'MaxSharpeRatio',
        'percent': 0.1,
        'risk_aversion': 1.0
    },#优化参数 如已有权重则不起作用
    TP_SL={
        'method': 'fixed_ratio',
        'up_pct': 0.1,
        'down_pct': 0.05
    }#止盈止损参数 可选
)

# 运行回测
backtest.run_backtest()

# 展示结果
backtest.plot_results()

# 保存结果
backtest.save('my_strategy_results.csv')
```

### 单因子测试

```python
from factor_testing import single_factor_test

single_factor = single_factor_test(
    num_layer = 5, 
    end_date = None,
    start_date = None,
    periods= (1,5,10,20), # 周期分别对应日频周频月频
    factor_df = pd.DataFrame(df_lstm.iloc[:,0]) , #因子数据dataframe 要求双索引 level0是date level1是stock_code 有唯一的因子值列
    close_df = df_close  ,# 股票收盘价数据 要求列名形式为'000000.XSHG'（与rebalance_df保持一致） index为‘date’ 建议后向填充一下 不填充的话更真实
    percent_ls = 0.1,
    factor_name= 'LSTM')
single_factor.show_performance()
```

---

## 未来改进方向

- **引入更多优化算法**：支持机器学习和深度学习方法的组合优化。
- **完善交易成本模型**：考虑滑点和市场冲击，提升回测的真实度。
- **丰富风险管理策略**：增加动态止盈止损、风险预算等功能。
- **增强数据兼容性**：支持更多数据格式和来源。
- **优化计算性能**：利用 GPU 加速和更多并行技术，提升运行速度。
- **模块化设计**：提高代码的可扩展性，方便用户二次开发。

---

## 总结

本项目整合了回测框架、投资组合优化器和单因子测试三个核心模块，提供了一套高效、易用的量化投资工具。用户仅需提供必要的数据，即可完成从策略生成、优化到回测的完整流程。通过支持并行计算和丰富的功能配置，框架在保证运行速度的同时，提供了灵活性和可扩展性，助力用户更快地实现策略开发与验证。

---
