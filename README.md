### June 2023 Rebase

## Main Goal: Perform backtesting of model, and be confident in validity of test (no test data leaks)

Background: Back during the project, we manually gathered some over unders and measured the models performance,
and the model did very well. However, there were only maybe 60 games used. Want more thorough testing.

Good source for betting lines to measure performance:
https://www.reddit.com/r/sportsbook/comments/rslmm3/database_of_nba_spreadsovers_and_almost_all_box/

Steps:

- Relearn codebase and make sure things are still working (API, tables, etc)
- Create comprehensive and easy to use testing functionality (and ensure no data leaks)
- Load all betting lines data into system
- Think carefully about how to select hyperparams using val data to avoid upward biasing test acc estimate
- Get test acc estimate