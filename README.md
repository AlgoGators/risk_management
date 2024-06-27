# Risk Management and Dynamic Optimization

## Risk Management

### What is Risk Management?
---
Risk Management is tasked with three objectives:
1. Identify current market risk levels to be used as inputs (under risk_measures)
2. Systematically reduce individual position and total portfolio risks (under risk_limits)
3. Automatically alert of changing market conditions that need action (under risk_reporting)

### Goals:
- [x] Integrate risk_limits into Dynamic Optimization
- [x] Effectively record whenever a risk limit is activated
- [ ] Move the portfolio statistical calculations to risk_measures and then reference them in risk_limits (helping us see how these values change over time)
- [ ] Asymmetric measures of risk for equities
- [ ] Build out automated reporting
- [ ] Develop different VaR models like MVaR, IVaR as well as proper limits
- [ ] Use different distributions that may be more accurate for predictions
- [ ] Improve GARCH (weights for each instrument?)
- [ ] Look into using Integrated GARCH model for VaR estimate (used by JPM and includes a unit root—volatility doesn't mean revert).

#### GARCH (and variations):
- [x] GARCH
- [ ] IGARCH – for VaR estimates
- [ ] TGARCH - for equity markets (asymmetric variances)

## Dynamic Optimization

### What is Dynamic Optimization?
---
Dynamic optimization seeks to find the set of integer positions that minimizes the standard deviation of tracking error against the ideal float positions. 

### Goals
- [x] Integrate risk limits into the algorithm
- [ ] Build out robust unittesting to help identify and mitigate any potential rooms for propegating data errors
- [x] Rewrite the algorithm (maybe using simulated annealing) to increase efficiency...
- [ ] **OR** switch to a faster programming language such as Rust or C++ 

## Building Docker Containers
Building dyn_opt:
- Open command line in the parent directory (risk_management) and run:

```docker build -f docker/Dockerfile.dyn_opt -t dyn_opt_container .```

Building risk_measures:
- ...

```docker build -f docker/Dockerfile.risk_measures -t risk_measures_container .```


## Running Docker Containers
Running dyn_opt:
- Open command line and run:

```docker run -p 5000:5000 dyn_opt_container```

Running risk_measures:

```docker run -p 5001:5001 risk_measures_container```