#!/bin/bash

frame=0

# 480p
#W=640
#H=480

# 1080p
W=1920
H=1080

y=40
z=40

POVDIR=pov

###########################

rm -rf $POVDIR
mkdir  $POVDIR
cp ../render.top render.sge

for x in {30..50} 
do 
  for _x in {0..9}
  do
    i=`echo $frame | awk '{printf("%05d", $1)}'`
    povheader -standard_light -ambient_light 3 3 3 -camera $x.$_x $y $z > $POVDIR/${i}.pov
    fvi2pov < ../../PS_1.fvi256 >> $POVDIR/${i}.pov &
    echo "FRAME:	$frame"
    echo "povray -d -W$W -H$H ${i}.pov &" >> render.sge
    frame=`expr ${frame} + 1`
sleep 2
  done
done

echo "Waiting..."

wait

mail frankwillmore@gmail.com < $DIR/render.sge

