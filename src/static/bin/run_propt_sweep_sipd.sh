#!/bin/bash
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=simplex ++propt.reduce=player ++propt.index=0 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=simplex ++propt.reduce=player ++propt.index=1 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=simplex ++propt.reduce=player ++propt.index=2 ++thread_pool.size=6 
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=simplex ++propt.reduce=player ++propt.index=3 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=simplex ++propt.reduce=player ++propt.index=4 ++thread_pool.size=6 

python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=softmax ++propt.reduce=player ++propt.index=0 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=softmax ++propt.reduce=player ++propt.index=1 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=softmax ++propt.reduce=player ++propt.index=2 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=softmax ++propt.reduce=player ++propt.index=3 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo_ordered ++propt.method=nowak_radzik ++propt.proj=softmax ++propt.reduce=player ++propt.index=4 ++thread_pool.size=6

python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=0 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=1 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=2 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=3 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=4 ++thread_pool.size=6

python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=softmax ++propt.reduce=player ++propt.index=0 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=softmax ++propt.reduce=player ++propt.index=1 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=softmax ++propt.reduce=player ++propt.index=2 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=softmax ++propt.reduce=player ++propt.index=3 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=softmax ++propt.reduce=player ++propt.index=4 ++thread_pool.size=6

python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=0 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=1 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=2 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=3 ++thread_pool.size=6
python src/static/propt.py +xeval=sipd_ppo ++propt.method=shapley ++propt.proj=simplex ++propt.reduce=player ++propt.index=4 ++thread_pool.size=6