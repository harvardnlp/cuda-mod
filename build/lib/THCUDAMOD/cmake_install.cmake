# Install script for directory: /n/rush_lab/users/yoonkim/cuda-mod/lib/THCUDAMOD

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/n/home01/yoonkim/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib/libTHCUDAMOD.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib/libTHCUDAMOD.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib/libTHCUDAMOD.so"
         RPATH "$ORIGIN/../lib:/n/home01/yoonkim/torch/install/lib:/n/rush_lab/sw/usr/local/cuda-7.0/lib64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib" TYPE MODULE FILES "/n/rush_lab/users/yoonkim/cuda-mod/build/lib/THCUDAMOD/libTHCUDAMOD.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib/libTHCUDAMOD.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib/libTHCUDAMOD.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib/libTHCUDAMOD.so"
         OLD_RPATH "/n/home01/yoonkim/torch/install/lib:/n/rush_lab/sw/usr/local/cuda-7.0/lib64:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/n/home01/yoonkim/torch/install/lib:/n/rush_lab/sw/usr/local/cuda-7.0/lib64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/n/rush_lab/sw/x86_64-unknown-linux-gnu/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/cuda-mod/1.0-0/lib/libTHCUDAMOD.so")
    endif()
  endif()
endif()

