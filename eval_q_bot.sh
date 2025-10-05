#!/bin/zsh

learning_agent=agents/policy_agent_9_initial/policy_agent_initial.hdf5
#learning_agent=agents/policy_agent_9_1/policy_agent_1_20250803_lr_00100_bs_128.hdf5

learning_rates=("10000" "01000" "00100" "00010" "00001")
batch_sizes=(128 256 512 1024 2048 4096)

function convert {
  local lr="$1"
  case "$lr" in
    "10000")
      print "1.0"
    ;;
    "01000")
      print "0.1"
    ;;
    "00100")
      print "0.01"
    ;;
    "00010")
      print "0.001"
    ;;
    "00001")
      print "0.0001"
    ;;
    *)
      print "0"
    ;;
  esac
}

for lr in $learning_rates
do
  for bs in $batch_sizes
  do
    trained_agent="agents/policy_agent_9_2_lr_${lr}_bs_${bs}.hdf5"
    echo "Evaluating... ${trained_agent}"
    python eval_pg_bot.py --agent1 $learning_agent --agent2 $trained_agent --num-games 100
    echo "Done with learning_rate=$(convert $lr) and batch_size=$bs"
    echo ""
  done
done
