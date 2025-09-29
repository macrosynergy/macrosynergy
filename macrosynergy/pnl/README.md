# `macrosynergy.pnl`

Subpackage for calculating Profits and Loss (PnL) of positions (portfolios).

## Naive PnL

The module `naive_pnl.py` is the entry point. It is run using the `NaivePnL` class. The class is designed to facilitate PnL estimation without taking into account position sizes, transaction costs, or leverage rules. It is a simple way to estimate PnLs to test out trading strategies or to illustrate the potential PnLs of a set of trading signals.

```{warning}
Proxy PNL is currently an experimental functionality and is in beta-testing. It should not be used in production.
```

## Proxy PnL

With the entry point to this module being the `ProxyPnL` class, the purpose of this set of functions or class is to facilitate PnL estimation under the consideration of

1. assets under management (AUM),
2. volatility target or leverage rule,
3. transaction costs.

Like `NaivePnL` (in `naive_pnl.py`) this class works with a set of cross-section-specific signals and produces an illustrative PnL, but to consider position sizes and contract-specific transaction costs it proceeds differently.

In particular, the class allows proceeding in three separate steps, implemented through three different base methods:

- The method `contract_signals` transforms standard cross-sectional trading signals into notional contract-specific position signals. For example, if the raw signals refer to vol-targeted positions, the signals need to be inversely proportional to estimated volatility. Or if the signal refers to a hedged position with a hedge basket, the signal needs to be translated into positions for the main contract and all contracts that make up the hedging basket, in proportion to the hedge ratio and the basket shares (under consideration of potential contract blacklisting). Finally, the position signals need to be consolidated so that there is only one position signal per contract.

- The method `notional_positions` transforms contract signals into USD positions. This can be done with two principal methods. The first is to assign an AUM number and a leveraging rule. This means that the total sum of the positions is limited to a specified multiple of the set AUM. The second principal method targets the expected volatility of the portfolio based on the estimated volatility of the contract returns and their correlation. The second method will require an estimate_portfolio_vol function that estimates portfolio volatility based on contract sizes and historical returns.

- The method `proxy_pnl` multiplies positions with proxy returns and estimates transaction costs. In particular, it uses a trading_cost method to apply transaction costs and their size dependency to discretionary position changes. And it applies a roll_cost method to apply roll costs to positions at certain intervals. This function also should provide some analytics as to estimated PnLs across sections and the impact of trading costs.

### Terminology

To calculate proxy PnL we define our signals (and returns) into three different stages:

1. Risk signals and returns: not tradable assets (purely theoretical construct), but idiosyncratic property makes it attractive for signal construction.
2. Contract signal: tradable asset position implied, but only relative sizing between the different instruments is defined as it has no absolutely scale.
3. Notional position: scale is defined as either a leverage rule of AUM allocation or a volatility target in terms of USD.

### Flow

The typical flow when using the `ProxyPnL` class is as follows:

1. Calculating `contract_signals` from cross-sectional hedge ratios and trading signals.
2. Calculating `notional_positions` from contract signals and `AUM` or `volatility_target`.
3. (Download transaction statistics into a `QuantamentalDataFrame`)
4. Calculating `proxy_pnl` from notional positions and transaction costs.

### Flowchart and Diagrams

#### Key

In this example:

- The shapes of each elements is indicative of the type of entity they represent.

  - Hexagonal boxes represent outputs.
  - Rounded boxes represent functions.
  - Rectangular boxes represent inputs.
  - "Cylindrical" boxes represent data stores (dataframes, disk storage, etc.).

- `function` directly outputs to `direct output`.
- `function` internally calls `backend function`.
- `backend function` internally outputs `direct output`.
- Any interactions represented by dotted lines are not directly exposed to the user.

```{mermaid}
flowchart LR
  A[input]-->B([function])
  B --> C{{output}}
  B -.-> D([backend function])
  D -.-> C

```

#### Contract Signals Flow

```{mermaid}

flowchart TD;
    Sig[Signals]
    CSCALES[Contract Specific Scaling]
    HR[Hedge-Ratios]
    HB[Hedge Basket]
    CSfunc(['contract_signals'])
    CSOutput{{Contract Signals}}

    subgraph CS_UINP[User Inputs]
      Sig
      HR
      HB
      CSCALES
    end

      CS_UINP --> CSfunc
    CSfunc --> CSOutput
```

#### Notional Positions Flow

```{mermaid}
flowchart TD
  CS{{Contract Signals}}
  AUM[AUM]
  LVG[Leverage]
  LVGfunc['leverage_positions']
  VT[Volatility Target]
  HPVfunc(['historical_portfolio_volatility'])
  VTPOSfunc(['volatility_target_positions'])
  NP{{Notional Positions}}
  HPV{{Historical Portfolio Volatility}}
  VCV{{Variance-Covariance Matrices}}
  ESTF[Estimation Frequencies]
  ESTW[Weights per Estimation Frequency]
  NPfunc(['notional_positions'])
    VT_UINP -.-> VTPOSfunc
    LVG -.-> LVGfunc


    subgraph NP_UINP[User Inputs]
      LVG
      AUM
      CS
      subgraph VT_UINP[Volatility Target Specific Inputs]
        VT
        ESTF
        ESTW
      end
    end
    subgraph NP_METHODS[Methods]
      NPfunc

      LVGfunc
      VTPOSfunc
      HPVfunc
    end

    subgraph OUTPUTS[Outputs]
      NP
      HPV
      VCV
    end
    NPfunc -.-> VTPOSfunc

    HPVfunc --> HPV
    HPVfunc --> VCV
    NPfunc -.-> LVGfunc
    VTPOSfunc <-.-> HPVfunc

    LVGfunc -.-o NP
    NPfunc --> NP
    VTPOSfunc -.-o NP
    NP_UINP --> NPfunc
```

#### Proxy PnL Flow

```{mermaid}
flowchart TD
  PPfunc([Proxy PnL])
  NP{{Notional Positions}}
  PNLx{{PnL with Transaction Costs}}
  PNLr{{PnL without any Costs}}
  TCS{{Position-Specific Transaction Costs}}
  TCObj[`TransactionCost` Object]

  subgraph PP_UINP[User Inputs]
    NP
    TCObj
  end

  NP --> PPfunc
  TCObj --> PPfunc

  subgraph OUTPUTS[Outputs]
    PNLx
    PNLr
    TCS
  end
  PPfunc --> PNLx
  PPfunc --> PNLr
  PPfunc --> TCS

```

### Transaction Costs Object

```{mermaid}
flowchart TD


  subgraph TCObj[`TransactionCost` Object]
    TCDownload([Download Transaction Statistics]) --> CostDF[(Costs DataFrame)]


    subgraph TCINTMETHODS[Internal Working]
      CostDF -.-> TCExtrapolate([Extrapolate Transaction Statistics])
    end
    TCExtrapolate -.-> TCBidOffer([Calculate Bid-Offer Spread])
    TCExtrapolate -.-> TCRoll([Calculate Roll Costs])
  end
  POSITION[Postion i.e. Trade Size, Cross Section, Date]
  BIDOFFER{{Bid-Offer Spread}}
  ROLLCOST{{Roll Costs}}
  POSITION --> TCBidOffer --> BIDOFFER
  POSITION --> TCRoll --> ROLLCOST

```
