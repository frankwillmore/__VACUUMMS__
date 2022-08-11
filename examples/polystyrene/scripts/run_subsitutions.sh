for i in 12 20 63 70 71 72
do
    export RESULT=$(grep "^$i " pair.dat | awk '{print $3"\t"$2}')
    export VAL_${i}="${RESULT}"
#    VAL_${i}=$(grep "$i " pair.dat | awk '{print $3"\t"$2}')
#    echo "VAL_${i} = ${RESULT}"
done


while read -r -a array
do 
    atom_type=${array[0]}

    [ ${atom_type} == "12" ] && printf "${array[1]}\t${array[2]}\t${array[3]}\t${VAL_12}\n"
    [ ${atom_type} == "20" ] && printf "${array[1]}\t${array[2]}\t${array[3]}\t${VAL_20}\n"
    [ ${atom_type} == "63" ] && printf "${array[1]}\t${array[2]}\t${array[3]}\t${VAL_63}\n"
    [ ${atom_type} == "70" ] && printf "${array[1]}\t${array[2]}\t${array[3]}\t${VAL_70}\n"
    [ ${atom_type} == "71" ] && printf "${array[1]}\t${array[2]}\t${array[3]}\t${VAL_71}\n"
    [ ${atom_type} == "72" ] && printf "${array[1]}\t${array[2]}\t${array[3]}\t${VAL_72}\n"

done < PS_Corrected.atoms

# Re-pack the simulation box to shift it to start at origin
cram -box 87.57884525964772 87.57884525964772 87.57884525964772 < PS_Corrected.gfg > PS_Corrected_crammed.gfg
