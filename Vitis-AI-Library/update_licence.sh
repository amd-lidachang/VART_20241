##
##  Copyright 2022-2023 Advanced Micro Devices Inc.
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##  http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##

c_start_py='## '
c_py='## '
c_end_py='## '

c_start_cpp='/* '
c_cpp=' * '
c_end_cpp='**/'

c_start_hpp='/* '
c_hpp=' * '
c_end_hpp='**/'

c_start_sh='## '
c_sh='## '
c_end_sh='## '

for ext in sh py cpp hpp; do
    for file in $(grep "Copyright [0-9]* Xilinx Inc." -r -i -L --include='*.'$ext); do
        content=$(cat $file)
        v=c_start_$ext
        start=${!v}
        v=c_end_$ext
        end=${!v}
        v=c_$ext
        comment=${!v}
        lic=$(while read -r i; do
                  echo "$comment $i";
              done < license.txt)

        cat  <<EOF >$file
${start}
${lic}
${end}
$content
EOF

    done
done
