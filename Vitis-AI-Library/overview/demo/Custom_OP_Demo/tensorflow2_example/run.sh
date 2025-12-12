cd op_Mylayer

rm -rf libvart_op_imp_Mylayer.so

make > /dev/null 2>&1

cp libvart_op_imp_Mylayer.so /usr/lib

cd ../tf2_custom_op_graph_runner

bash -x build.sh > /dev/null 2>&1

./tf2_custom_op_graph_runner /usr/share/vitis_ai_library/models/tf2_custom_op/tf2_custom_op.xmodel ../sample.jpg
