import sys
import re
output = sys.argv[1]
input = sys.argv[2]


hook_sets = set()
input_line = open(input, "r").readlines()
pattern = r"cpu_runer_reg_func_handle[a-zA-Z_0-9]+_hook"
hook_sets = set()
for line in input_line:
    matches = re.findall(pattern, line)
    hook_sets.update(matches)

hook_sets = sorted(hook_sets)

output = open(output, "w")
output.write("typedef void* void_ptr_t;\n")
for hook in hook_sets:
    output.write('extern "C" void* ' + hook + ';\n')


output.write("void_ptr_t vart_cpu_runner_reg_hooks[] = {\n")
for hook in hook_sets:
    output.write("\t" + hook + ",\n")
output.write("};\n")
