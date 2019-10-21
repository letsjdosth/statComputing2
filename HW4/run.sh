#!/bin/bash

count=$1

if [ "$count" == "" ]
then
	echo "error"
	exit 0
fi


while [ $count -gt 0 ]
do
	python c:/gitProject/statComputing2/HW4/HW4_item_resp_model_MCMC.py &
	echo $!
	count=`expr $count - 1`
done


