conda activate cs703
## declare an array variable
declare -a arr=("0" "-0.1" "-0.2" "-0.3")

## now loop through the above array
for i in "${arr[@]}"
do
   python3 appropo.py --init_variable both --prob_failure $i --threshold 0.12
done

## declare an array variable
declare -a arr=("0" "0.06" "0.12" "0.3")

## now loop through the above array
for i in "${arr[@]}"
do
   python3 appropo.py --init_variable both --prob_failure -0.2 --threshold $i

done