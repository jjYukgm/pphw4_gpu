#/bin/bash
stuID=`echo $USER`

function judge() {
	local ind=$3
	local ind2=""
	if [ "${2}" == "o" ]; then
	local ind2="$3"
	fi
	
	#HW4_105061516_cuda HW4_105061516_mpi HW4_105061516_openmp
	#job_Str_p job_Str_ms job_Str_ma
	job_id=$(echo $(qsub -v exe="HW4_105061516_${1}${ind2}.exe" "tm/job_$2_$1${ind}.sh") | awk -F '[. ]' '{print $3}')

	#esp esms esma
	while [ ! -f "e$2$1${ind}" ]; do
		sleep 0.2
	done
	sleep 0.2

	index_=$(printf '%2d' $ind)

	
	echo -e "TimeMeasure $index_ \E[0;32;40maccepted\E[0m"
}
# b w o
k=b
mi=1 #w 1-4, b 1-5, o 2-4
mx=4
if [ "${k}" == "o" ]; then
mi=2
elif [ "${k}" == "b" ]; then
mx=5
fi

# cuda openmp mpi
i=cuda
for ((j=${mi}; j<=${mx}; j=j+1)) 
do
	judge ${i} ${k} ${j}

done
# cuda openmp mpi
i=openmp
for ((j=${mi}; j<=${mx}; j=j+1)) 
do
	judge ${i} ${k} ${j}

done
# cuda openmp mpi
i=mpi
for ((j=${mi}; j<=${mx}; j=j+1)) 
do
	judge ${i} ${k} ${j}

done

#qsub tm2/job_p_p.sh
#while [ ! -f "epp" ]; do
#	sleep 0.2
#done
#sleep 0.2
#echo -e "TimeMeasure Pthread \E[0;32;40maccepted\E[0m"
