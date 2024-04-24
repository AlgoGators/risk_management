# Risk Management and Dynamic Optimization

## Risk Management

### What is Risk Management?
---
Risk Management is tasked with three objectives:
1. Identify current market risk levels to be used as inputs (under risk_metrics)
2. Systematically reduce individual position and total portfolio risks (under risk_limits)
3. Automatically alert of changing market conditions that need action (under risk_reporting)

### Goals:
- [ ] Effectively record whenever a risk limit is activated
- [ ] Move the portfolio statistical calculations to risk_metrics and then reference them in risk_limits (helping us see how these values change over time)
- [ ] Asymmetric measures of risk for equities
- [ ] Build out automated reporting
- [ ] Develop different VaR models like MVaR, IVaR as well as proper limits
- [ ] Integrate risk_limits into Dynamic Optimization
- [ ] Use different distributions that may be more accurate for predictions
- [ ] Improve GARCH

## Dynamic Optimization

### What is Dynamic Optimization?
---
Dynamic optimization seeks to find the set of integer positions that minimizes the standard deviation of tracking error against the ideal float positions. 

### Goals
- [ ] Build out robust unittesting to help identify and mitigate any potential rooms for propegating data errors
- [ ] Integrate risk limits into the algorithm
- [ ] Rewrite the algorithm (maybe using simulated annealing) to increase efficiency...
- [ ] **OR** switch to a faster programming language such as Rust or C++ 
