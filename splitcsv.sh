#!/bin/bash
join -j 1 -t, transcription.csv $1 > ${1}_full_output.csv
OUTDIR=split_${1%.*}
mkdir ${OUTDIR}; 
for record in $(cut -d_ -f1,2 ${1}_full_output.csv  | sort -Vu); do 
	echo $record;
	grep ^${record}_ ${1}_full_output.csv > ${OUTDIR}/${record}_output.csv; 
done 
