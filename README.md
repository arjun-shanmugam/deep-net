# June 2023 Rebase

## Main Goal: Perform backtesting of model, and be confident in validity of test (no test data leaks)

Background: Back during the project, we manually gathered some over unders and measured the models performance,
and the model did very well. However, there were only maybe 60 games used. Want more thorough testing.

Good source for betting lines to measure performance:
https://www.reddit.com/r/sportsbook/comments/rslmm3/database_of_nba_spreadsovers_and_almost_all_box/

Steps/Checklist:

- [x] Review codebase
- [x] Check that APIs, tables still working
- [x] Port code to pytorch?
- [ ] Dive into data - see what we have and what's missing. What is supposed to be in games_and_players.dta?
- [ ] Find previously used hyperparameters
- [ ] Train a model and make sure it looks ok.
- [ ] Refactor Codebase (make it easy to interact with from commandline w/o code changes)
- [ ] Create comprehensive and easy to use testing functionality (and ensure no data leaks)
- [ ] Load all betting lines data into system
- [ ] Think carefully about how to select hyperparams using val data to avoid upward biasing test acc estimate
- [ ] Get test acc estimate

Stretch Goals:

- [ ] Investigate more optimal ways to allocate betting money based on confidence
- [ ] Implement allocation strategy and test
- [ ] Automate prediction serving (loading lines automatically)