#/bin/bash

echo "Generate the job file...."
# 輸出檔名, 數量
# %1 = core數量

# w b o
f0=o
# cuda openmp mpi
fn0=cuda
# job_w_ job_b_ job_o_
fn1='job_'${f0}'_'${fn0}
# q001_1000 q002_2000 q003_4000 q004_8000
fn5=oott
#cd tm/
squ=1
for i in $(seq 1 $1) 
do
fn2=${fn1}$i'.sh'
# ewc ewm ewo ebc ebo ebm
fn4=e${f0}${fn0}$i
# owc owm owo out

if [ "${f0}" == "w" ]; then
fn5=o${f0}${fn0}$i # w
squ=$((2**$i))		# Block
te2=" q00$i_${squ}000 o51 32"
elif [ "${f0}" == "b" ]; then
squ=$((${squ}*2))	# Block
te2=" q004_8000 o51 ${squ}"
else
te2=" q004_8000 o51 32"
fi


echo "#/bin/bash													 "	>>${fn2}
echo "#PBS -N ${fn1}$i												 "	>>${fn2}
echo "#PBS -r n                                                      "	>>${fn2}

if [ "${fn0}" == "cuda" ]; then
echo "#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process                 "	>>${fn2} # cuda
else
echo "#PBS -l nodes=1:ppn=2:gpus=2:exclusive_process                 "	>>${fn2} # else
fi

echo "#PBS -l walltime=00:05:00                                      "	>>${fn2}
echo "#PBS -e ${fn4}	                                             "	>>${fn2}
echo "#PBS -o ${fn5}                                                 "	>>${fn2}
echo 'cd $PBS_O_WORKDIR                                              '	>>${fn2}
echo "export OMP_NUM_THREADS=2                                       "	>>${fn2}

if [ "${fn0}" == "mpi" ]; then
te1='time mpirun ' # mpi
else
te1='time '
fi
echo "${te1}"'./$exe'"${te2}"                        							""	>>${fn2}
# ${exe}'"$i.exe q004_16000 o51 32 # otps
# q00$i_${squ}000 o51 32	# weak
# q004_16000 o51 ${squ}		# block

done

exit 0
