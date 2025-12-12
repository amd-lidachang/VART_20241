/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOG_H
#define LOG_H

#include <cstdio>

#define LOG_LEVEL_ERROR 2
#define LOG_LEVEL_DEBUG 1
#define LOG_LEVEL_NOLOG 0
#define LOG_LEVEL 2

#define RED "\e[1;31m"
#define GREEN "\e[1;32m"
#define NONE "\e[0m"

#define logd(fmt, ...)                                                         \
  printf((GREEN "[DEBUG]" NONE "[%s]:" fmt "\n"), __FUNCTION__, ##__VA_ARGS__)
#define loge(fmt, ...)                                                         \
  if (LOG_LEVEL > 1)                                                           \
  printf((RED "[ERROR]" NONE "[%s][%s][line:%d]:" fmt "\n"), __FILE__,         \
         __FUNCTION__, __LINE__, ##__VA_ARGS__)

template <typename T0, typename T>
void print_n_int8(T In, int n, char* s) {
  //	logd("%s", s);
  T0* p = (T0*)(&In);
  for (int i = 0; i < n; i = i + 8) {
    printf("%s: %d: %d, %d, %d, %d, %d,%d, %d, %d\n", s, i / 8 + 1, *(p),
           *(p + 1), *(p + 2), *(p + 3), *(p + 4), *(p + 5), *(p + 6),
           *(p + 7));
    p = p + 8;
  }
}

template <typename T0, typename T>
void print_n_int8(T* In, int n, char* s) {
  logd("%s", s);
  T0* p = (T0*)(In);
  for (int i = 0; i < n; i = i + 8) {
    printf("%s: %d: %d, %d, %d, %d, %d,%d, %d, %d\n", s, i / 8 + 1, *(p),
           *(p + 1), *(p + 2), *(p + 3), *(p + 4), *(p + 5), *(p + 6),
           *(p + 7));
    p = p + 8;
  }
}

#endif
