#! /bin/bash
order=4
for i in 8e-2 6e-2 4e-2 ; do 
  gmsh -2 -order $order \
    -string "Mesh.CharacteristicLengthMax = $i;" \
    -string "Geometry.OCCTargetUnit='MM';" \
    -format msh2 \
    -o blob2d-order$order-h$i.msh \
    blob-2d.step
done
