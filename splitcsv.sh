#!/bin/bash
#Add transcriptions from transcriptions.csv (output from ICDAR'17 Method)
join -j 1 -t, transcription.csv $1 > tmpfile1_$$
#Remove words with no category,track complete
grep -v ',other,' tmpfile1_$$ > tmpfile2_$$ 
#Remove words with no person
grep -v ',none$' tmpfile2_$$ > tmpfile3_$$ 
#Remove words with no category, if file is only track basic
grep -v ',other$' tmpfile3_$$ > ${1%.*}_full_output.csv 
OUTDIR=split_${1%.*}
mkdir ${OUTDIR}; 
for record in $(cut -d_ -f1,2 ${1%.*}_full_output.csv  | sort -Vu); do 
	echo $record;
	grep ^${record}_ ${1%.*}_full_output.csv > ${OUTDIR}/${record}_output.csv; 
done 
rm  tmpfile?_$$ ${1%.*}_full_output.csv 
