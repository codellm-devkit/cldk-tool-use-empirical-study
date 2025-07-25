# sweagent run \
#   --agent.model.name=deepseek/deepseek-chat \
#   --agent.model.per_instance_cost_limit=2.00 \
#   --env.repo.github_url=https://github.com/SWE-agent/test-repo \
#   --problem_statement.github_url=https://github.com/SWE-agent/test-repo/issues/22

sweagent run-batch \
    --config config/structural_search.yaml \
    --agent.model.name deepseek/deepseek-chat \
    --num_workers 3 \
    --agent.model.per_instance_cost_limit 2.00 \
    --instances.deployment.docker_args=--memory=10g \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test  \
    --instances.slice 301:302 \
    --instances.shuffle=False \