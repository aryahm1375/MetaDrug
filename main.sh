
#export PYTHONPATH=/home/a053h213/PycharmProjects/pythonProject/:$PYTHONPATH

for DIM in 256; do
    for seed in 0; do
      python main.py --cuda 1 --seed $seed --dim $DIM --title "main" --visit --user
    done
done

