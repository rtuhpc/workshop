count=`qstat -t -f $1[] | grep job_state | wc -l`
count_finish=0
while [ $count_finish -le $count ]
do 
#clear
count_finish=`qstat -t -f $1[] | grep job_state | grep C | wc -l`
echo -e "\n"
echo "Finished jobs: $count_finish from $count"
sleep 5
done
#convert -delay 20 $2/rend*.jpg $2.gif
