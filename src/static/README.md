# Static Analysis of Cooperative Curriculum Learning


## Experiments


### Shapley-Proportional Curriculum

#### Sparse IPD

```
    python propt.py +xper=sipd_ppo ++propt.method=shapley
```

```
    python propt.py +xper=sipd_ppo_ordered ++propt.method=nowak_radzik
```

```
    python propt.py +xper=sipd_ppo_ordered ++propt.method=sanchez_bergantinos
```

#### MinAtar

```
    python propt.py +xper=minatar_dqn ++propt.method=shapley ++run.log_every=50000
```

```
    python propt.py +xper=minatar_dqn_ordered ++propt.method=nowak_radzik ++run.log_every=50000
```

```
    python propt.py +xper=minatar_dqn_ordered ++propt.method=sanchez_bergantinos ++run.log_every=50000
```

