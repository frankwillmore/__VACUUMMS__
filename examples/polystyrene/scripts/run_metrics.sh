#!/bin/bash

# read the .cls file for the sample and extract statistics for each of the clusters

box_dims=" -box 87.57884525964772 87.57884525964772 87.57884525964772 "

infile=PS_1M.cls
cp /dev/null PS_1M_end2end.dst 
cp /dev/null PS_1M_rog.dst 
cp /dev/null PS_1M_cv.dst 
cp /dev/null PS_1M_csa.dst 

n_clusters=$(tail -n 1 < ${infile} | awk '{print $1}') 

echo got n_clusters=$n_clusters

for cluster in $(seq 0 ${n_clusters})
do
    echo "processing cluster #${cluster}"

    # One at a time, pull individual clusters and extract metrics
    grep -P "^${cluster}\t" \
        < PS_1M.cls \
        | awk '{print $3"\t"$4"\t"$5"\t"$6}' \
        | center \
            ${box_dims} \
        > cls.cav
#    wc cls.cav

    # calculate the span of each cluster
    end2end \
        ${box_dims} \
        < cls.cav \
            >> PS_1M_end2end.dst 

    # calculate the rog of each cluster
    rog \
        ${box_dims} \
        < cls.cav \
            >> PS_1M_rog.dst 

    # calculate the rog of each cluster
    cv \
        ${box_dims} \
        < cls.cav \
            >> PS_1M_cv.dst 

    # calculate the rog of each cluster
    csa \
        ${box_dims} \
        < cls.cav \
            >> PS_1M_csa.dst 
done

echo "cluster span distribution:" 
dst2hst -width 0.5 < PS_1M_end2end.dst | tee PS_1M_end2end.hst

echo

echo "cluster rog distribution:"
dst2hst -width 0.1 < PS_1M_rog.dst  | tee PS_1M_rog.hst

echo

echo "cluster volume distribution:"
dst2hst -width 4.0 < PS_1M_cv.dst  | tee PS_1M_cv.hst

echo

echo "cluster surface area distribution:"
dst2hst -width 5.0 < PS_1M_csa.dst  | tee PS_1M_csa.hst

echo

echo "cavity size distribution:"
awk '{print $4}' < PS_1M.unq | dst2hst -width 0.125 | tee PS_1M.hst

echo

