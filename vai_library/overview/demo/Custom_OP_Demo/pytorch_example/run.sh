cd op_registration/
sh -x op_registration.sh
cd ../pointpillars_graph_runner
sh -x build.sh
./build/sample_pointpillars_graph_runner /usr/share/vitis_ai_library/models/pt_custom_op/pt_custom_op.xmodel data/sample_pointpillars.bin
